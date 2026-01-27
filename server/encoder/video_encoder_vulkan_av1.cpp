/*
 * WiVRn VR streaming
 * Copyright (C) 2026  BERADQ <adqber123@outlook.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
#include "video_encoder_vulkan_av1.h"

#include "encoder/encoder_settings.h"
#include "utils/wivrn_vk_bundle.h"

static StdVideoAV1Level compute_level(int width, int height, float fps, uint32_t num_dpb_frames, size_t bitrate)
{
	// Simple AV1 level selection based on resolution and bitrate
	const uint64_t pixel_rate = uint64_t(width) * uint64_t(height) * uint64_t(fps);

	if (pixel_rate <= 2073600) // Level 3.0 - up to 1280x720 @ 30fps
		return STD_VIDEO_AV1_LEVEL_3_0;
	if (pixel_rate <= 4177920) // Level 4.0 - up to 1920x1080 @ 30fps
		return STD_VIDEO_AV1_LEVEL_4_0;
	if (pixel_rate <= 9953280) // Level 5.0 - up to 1920x1080 @ 60fps
		return STD_VIDEO_AV1_LEVEL_5_0;
	if (pixel_rate <= 20736000) // Level 5.3 - up to 2560x1440 @ 60fps
		return STD_VIDEO_AV1_LEVEL_5_3;

	return STD_VIDEO_AV1_LEVEL_6_0; // Higher levels
}

wivrn::video_encoder_vulkan_av1::video_encoder_vulkan_av1(
        wivrn_vk_bundle & vk,
        const vk::VideoCapabilitiesKHR & video_caps,
        const vk::VideoEncodeCapabilitiesKHR & encode_caps,
        uint8_t stream_idx,
        const encoder_settings & settings) :
        video_encoder_vulkan(vk, video_caps, encode_caps, stream_idx, settings)
{
	if (not std::ranges::any_of(vk.device_extensions, [](const std::string & ext) { return ext == VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME; }))
	{
		throw std::runtime_error("Vulkan video encode AV1 extension not available");
	}

	// Initialize the AV1 sequence header optimized for ultra-low latency
	sequence_header = {};
	sequence_header.flags.still_picture = 0;
	sequence_header.flags.reduced_still_picture_header = 1; // Enable for faster processing
	sequence_header.flags.use_128x128_superblock = 0;       // Use 64x64 for lower latency
	sequence_header.flags.enable_filter_intra = 0;          // Disable for lower latency
	sequence_header.flags.enable_intra_edge_filter = 0;     // Disable for lower latency
	sequence_header.flags.enable_interintra_compound = 0;
	sequence_header.flags.enable_masked_compound = 0;
	sequence_header.flags.enable_warped_motion = 0; // Disable for lower complexity
	sequence_header.flags.enable_dual_filter = 0;
	sequence_header.flags.enable_order_hint = 0; // Disable for fastest frame processing
	sequence_header.flags.enable_jnt_comp = 0;
	sequence_header.flags.enable_ref_frame_mvs = 0; // Disable for lower latency
	sequence_header.flags.frame_id_numbers_present_flag = 0;
	sequence_header.flags.enable_superres = 0;
	sequence_header.flags.enable_cdef = 0;        // Disable CDEF for faster encoding
	sequence_header.flags.enable_restoration = 0; // Disable loop restoration for lowest latency
	sequence_header.flags.film_grain_params_present = 0;
	sequence_header.flags.timing_info_present_flag = 0;
	sequence_header.flags.initial_display_delay_present_flag = 0;

	sequence_header.seq_profile = settings.bit_depth == 10 ? STD_VIDEO_AV1_PROFILE_HIGH : STD_VIDEO_AV1_PROFILE_MAIN;
	sequence_header.frame_width_bits_minus_1 = 13; // Reduce range to 8192 (still sufficient for 8K)
	sequence_header.frame_height_bits_minus_1 = 13;
	sequence_header.max_frame_width_minus_1 = extent.width - 1;
	sequence_header.max_frame_height_minus_1 = extent.height - 1;
	sequence_header.delta_frame_id_length_minus_2 = 3; // Use minimal frame ID length for speed
	sequence_header.additional_frame_id_length_minus_1 = 0;
	sequence_header.order_hint_bits_minus_1 = 3; // Reduced for faster processing

	// Set up color configuration optimized for VR streaming
	StdVideoAV1ColorConfig color_config = {};
	color_config.flags.mono_chrome = 0;
	color_config.flags.color_range = 0; // Studio swing for compatibility
	color_config.flags.separate_uv_delta_q = 0;
	color_config.flags.color_description_present_flag = 0;
	color_config.BitDepth = settings.bit_depth;
	color_config.subsampling_x = 1; // 4:2:0 chroma subsampling (standard for video)
	color_config.subsampling_y = 1;
	color_config.chroma_sample_position = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_COLOCATED; // Colocated for faster processing
	color_config.color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709;
	color_config.transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SRGB; // sRGB for faster gamma processing
	color_config.matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_IDENTITY;       // Identity matrix for fastest color transform
	sequence_header.pColorConfig = &color_config;
}

std::vector<void *> wivrn::video_encoder_vulkan_av1::setup_slot_info(size_t dpb_size)
{
	dpb_std_info.resize(dpb_size, {});
	dpb_std_slots.reserve(dpb_size);
	std::vector<void *> res;
	for (size_t i = 0; i < dpb_size; ++i)
	{
		dpb_std_slots.push_back({
		        .pStdReferenceInfo = &dpb_std_info[i],
		});
		res.push_back(&dpb_std_slots[i]);
	}

	return res;
}

static auto get_video_caps(vk::raii::PhysicalDevice & phys_dev, int bit_depth)
{
	if (!(bit_depth == 8 || bit_depth == 10))
		throw std::runtime_error("AV1 encoder supports 8-bit or 10-bit only");

	vk::VideoProfileInfoKHR video_profile_info_value{};
	video_profile_info_value.videoCodecOperation = vk::VideoCodecOperationFlagBitsKHR::eEncodeAv1;
	video_profile_info_value.chromaSubsampling = vk::VideoChromaSubsamplingFlagBitsKHR::e420;
	video_profile_info_value.lumaBitDepth = bit_depth == 10 ? vk::VideoComponentBitDepthFlagBitsKHR::e10 : vk::VideoComponentBitDepthFlagBitsKHR::e8;
	video_profile_info_value.chromaBitDepth = bit_depth == 10 ? vk::VideoComponentBitDepthFlagBitsKHR::e10 : vk::VideoComponentBitDepthFlagBitsKHR::e8;

	vk::VideoEncodeAV1ProfileInfoKHR profile_info_value{};
	profile_info_value.stdProfile = bit_depth == 10 ? STD_VIDEO_AV1_PROFILE_HIGH : STD_VIDEO_AV1_PROFILE_MAIN;

	vk::VideoEncodeUsageInfoKHR usage_info_value{};
	usage_info_value.videoUsageHints = vk::VideoEncodeUsageFlagBitsKHR::eRecording;
	usage_info_value.videoContentHints = vk::VideoEncodeContentFlagBitsKHR::eRendered;
	usage_info_value.tuningMode = vk::VideoEncodeTuningModeKHR::eUltraLowLatency;

	vk::StructureChain video_profile_info{
	        video_profile_info_value,
	        profile_info_value,
	        usage_info_value};

	try
	{
		auto [video_caps, encode_caps, encode_av1_caps] =
		        phys_dev.getVideoCapabilitiesKHR<
		                vk::VideoCapabilitiesKHR,
		                vk::VideoEncodeCapabilitiesKHR,
		                vk::VideoEncodeAV1CapabilitiesKHR>(video_profile_info.get());
		return std::make_tuple(video_caps, encode_caps, encode_av1_caps, video_profile_info);
	}
	catch (...)
	{}
	// Some implementations fail if the structure is there
	video_profile_info.unlink<vk::VideoEncodeUsageInfoKHR>();

	auto [video_caps, encode_caps, encode_av1_caps] =
	        phys_dev.getVideoCapabilitiesKHR<
	                vk::VideoCapabilitiesKHR,
	                vk::VideoEncodeCapabilitiesKHR,
	                vk::VideoEncodeAV1CapabilitiesKHR>(video_profile_info.get());
	return std::make_tuple(video_caps, encode_caps, encode_av1_caps, video_profile_info);
}

std::unique_ptr<wivrn::video_encoder_vulkan_av1> wivrn::video_encoder_vulkan_av1::create(
        wivrn_vk_bundle & vk,
        const encoder_settings & settings,
        uint8_t stream_idx)
{
	if (settings.bit_depth != 8 && settings.bit_depth != 10)
		throw std::runtime_error("AV1 codec only supports 8-bit or 10-bit encoding");

	auto [video_caps, encode_caps, encode_av1_caps, video_profile_info] = get_video_caps(vk.physical_device, settings.bit_depth);

	std::unique_ptr<video_encoder_vulkan_av1> self(
	        new video_encoder_vulkan_av1(vk, video_caps, encode_caps, stream_idx, settings));

	// Initialize the sequence header first
	StdVideoAV1SequenceHeader av1_sequence_header = self->sequence_header;

	StdVideoEncodeAV1DecoderModelInfo decoder_model_info{};
	decoder_model_info.buffer_delay_length_minus_1 = 3; // Minimal delay for ultra-fast streaming
	decoder_model_info.buffer_removal_time_length_minus_1 = 15;
	decoder_model_info.frame_presentation_time_length_minus_1 = 15;
	decoder_model_info.num_units_in_decoding_tick = 1;

	StdVideoEncodeAV1OperatingPointInfo operating_point_info{};
	operating_point_info.flags.decoder_model_present_for_this_op = 0;
	operating_point_info.flags.low_delay_mode_flag = 1; // Enable low delay mode for VR
	operating_point_info.flags.initial_display_delay_present_for_this_op = 0;
	operating_point_info.operating_point_idc = 0; // Single operating point
	operating_point_info.seq_level_idx = compute_level(self->extent.width, self->extent.height, settings.fps, self->num_dpb_slots, settings.bitrate);
	operating_point_info.seq_tier = 0; // Main tier

	vk::VideoEncodeAV1SessionParametersCreateInfoKHR session_params_info{};
	session_params_info.pStdSequenceHeader = &av1_sequence_header;
	session_params_info.pStdDecoderModelInfo = &decoder_model_info;
	session_params_info.stdOperatingPointCount = 1;
	session_params_info.pStdOperatingPoints = &operating_point_info;

	vk::VideoEncodeAV1SessionCreateInfoKHR session_create_info{};

	// Configure rate control layer info - optimized for strict CBR to fill bitrate budget
	self->rate_control_layer_av1.useMinQIndex = true;
	self->rate_control_layer_av1.minQIndex.intraQIndex = 1; // Very low QP to avoid underusing bandwidth
	self->rate_control_layer_av1.minQIndex.predictiveQIndex = 1;
	self->rate_control_layer_av1.minQIndex.bipredictiveQIndex = 1;
	self->rate_control_layer_av1.useMaxQIndex = true;
	self->rate_control_layer_av1.maxQIndex.intraQIndex = 55; // Reasonable max QP for quality
	self->rate_control_layer_av1.maxQIndex.predictiveQIndex = 55;
	self->rate_control_layer_av1.maxQIndex.bipredictiveQIndex = 55;
	self->rate_control_layer_av1.useMaxFrameSize = true;                                            // Enable max frame size control for CBR
	self->rate_control_layer_av1.maxFrameSize.intraFrameSize = settings.bitrate / 8 / settings.fps; // Max frame size in bytes
	self->rate_control_layer_av1.maxFrameSize.predictiveFrameSize = settings.bitrate / 8 / settings.fps;
	self->rate_control_layer_av1.maxFrameSize.bipredictiveFrameSize = settings.bitrate / 8 / settings.fps;
	self->rate_control_layer.pNext = &self->rate_control_layer_av1;

	// Additional rate control info to enforce CBR behavior
	vk::VideoEncodeAV1RateControlInfoKHR rate_control_info{};
	rate_control_info.flags = vk::VideoEncodeAV1RateControlFlagBitsKHR::eTemporalLayerPatternDyadic; // Use temporal layers for better rate control
	rate_control_info.gopFrameCount = 0;                                                             // Open GOP for better rate control
	rate_control_info.keyFramePeriod = settings.fps * 10;                                            // Keyframes every 10 seconds to maintain quality
	rate_control_info.consecutiveBipredictiveFrameCount = 0;                                         // No B frames to reduce latency
	rate_control_info.temporalLayerCount = 1;                                                        // Single temporal layer for consistent quality

	self->init(video_caps, video_profile_info.get(), &session_create_info, &session_params_info);

	return self;
}

std::vector<uint8_t> wivrn::video_encoder_vulkan_av1::get_sequence_headers()
{
	vk::VideoEncodeSessionParametersGetInfoKHR next{};
	return get_encoded_parameters(&next);
}

void wivrn::video_encoder_vulkan_av1::send_idr_data()
{
	auto data = get_sequence_headers();
	SendData(data, false, true);
}

void * wivrn::video_encoder_vulkan_av1::encode_info_next(uint32_t frame_num_val, size_t slot, std::optional<int32_t> reference_slot)
{
	// Set up the AV1 picture info optimized for ultra-low latency
	std_picture_info = {};
	std_picture_info.flags.error_resilient_mode = 1; // Enable for robust VR streaming
	std_picture_info.flags.disable_cdf_update = 1;   // Speed up decoding
	std_picture_info.flags.use_superres = 0;
	std_picture_info.flags.render_and_frame_size_different = 0;
	std_picture_info.flags.allow_screen_content_tools = 0;
	std_picture_info.flags.is_filter_switchable = 0;
	std_picture_info.flags.force_integer_mv = 1;
	std_picture_info.flags.frame_size_override_flag = 0;
	std_picture_info.flags.buffer_removal_time_present_flag = 0;
	std_picture_info.flags.allow_intrabc = 0;
	std_picture_info.flags.frame_refs_short_signaling = 0;
	std_picture_info.flags.allow_high_precision_mv = 0; // Use lower precision for speed
	std_picture_info.flags.is_motion_mode_switchable = 0;
	std_picture_info.flags.use_ref_frame_mvs = 0;
	std_picture_info.flags.disable_frame_end_update_cdf = 1; // Speed up
	std_picture_info.flags.allow_warped_motion = 0;
	std_picture_info.flags.reduced_tx_set = 1;    // Use reduced transform set for speed
	std_picture_info.flags.skip_mode_present = 0; // Disable skip mode for consistent latency
	std_picture_info.flags.delta_q_present = 0;   // Disable delta Q for speed
	std_picture_info.flags.delta_lf_present = 0;
	std_picture_info.flags.delta_lf_multi = 0;
	std_picture_info.flags.segmentation_enabled = 0; // Disable segmentation for speed
	std_picture_info.flags.segmentation_update_map = 0;
	std_picture_info.flags.segmentation_temporal_update = 0;
	std_picture_info.flags.segmentation_update_data = 0;
	std_picture_info.flags.UsesLr = 0;
	std_picture_info.flags.usesChromaLr = 0;
	std_picture_info.flags.show_frame = 1;
	std_picture_info.flags.showable_frame = 1; // Always showable for consistent timing

	// Determine frame type based on reference availability but prioritize key frames for reliability
	constexpr int KEY_FRAME_INTERVAL = 60; // Force key frame every 60 frames for resilience
	bool is_key_frame = (frame_num_val % KEY_FRAME_INTERVAL) == 0 || !reference_slot;

	std_picture_info.frame_type = is_key_frame ? STD_VIDEO_AV1_FRAME_TYPE_KEY : STD_VIDEO_AV1_FRAME_TYPE_INTER;
	std_picture_info.frame_presentation_time = frame_num_val;
	std_picture_info.current_frame_id = frame_num_val;
	std_picture_info.order_hint = 0; // Set to 0 since we disabled order hints for speed
	std_picture_info.primary_ref_frame = is_key_frame ? STD_VIDEO_AV1_PRIMARY_REF_NONE : 0;
	std_picture_info.refresh_frame_flags = 0xFF; // Refresh all reference frames
	std_picture_info.coded_denom = 0;            // Full resolution
	std_picture_info.render_width_minus_1 = extent.width - 1;
	std_picture_info.render_height_minus_1 = extent.height - 1;
	std_picture_info.interpolation_filter = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP; // Use simple filter for speed
	std_picture_info.TxMode = STD_VIDEO_AV1_TX_MODE_LARGEST;                             // Use largest transform mode for speed
	std_picture_info.delta_q_res = 0;
	std_picture_info.delta_lf_res = 0;

	// Fill in reference info for P-frames
	if (!is_key_frame && reference_slot)
	{
		// Set up reference frames for inter-frame prediction
		for (int i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++)
		{
			std_picture_info.ref_frame_idx[i] = *reference_slot;
		}

		// Set up delta frame IDs
		for (int i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++)
		{
			std_picture_info.delta_frame_id_minus_1[i] = 1; // Minimal reference difference
		}
	}
	else
	{
		// Key frame: no references
		for (int i = 0; i < STD_VIDEO_AV1_REFS_PER_FRAME; i++)
		{
			std_picture_info.ref_frame_idx[i] = 0;
			std_picture_info.delta_frame_id_minus_1[i] = 0;
		}
	}

	// Initialize tile info for the frame - simplified for speed
	tile_info = {};
	tile_info.flags.uniform_tile_spacing_flag = 1;
	tile_info.TileCols = 2; // Use multiple tiles for parallel processing (adjust based on resolution)
	tile_info.TileRows = 2;
	tile_info.context_update_tile_id = 0;
	tile_info.tile_size_bytes_minus_1 = 0;

	// Set up quantization info optimized for speed and quality balance
	quantization = {};
	quantization.flags.using_qmatrix = 0;
	quantization.flags.diff_uv_delta = 0;
	quantization.base_q_idx = is_key_frame ? 15 : 25; // Lower QP for key frames to maintain quality
	quantization.DeltaQYDc = 0;
	quantization.DeltaQUDc = 0;
	quantization.DeltaQUAc = 0;
	quantization.DeltaQVDc = 0;
	quantization.DeltaQVAc = 0;
	quantization.qm_y = 15; // Default quantization matrix
	quantization.qm_u = 15;
	quantization.qm_v = 15;

	// Set up the standard picture info pointers
	std_picture_info.pTileInfo = &tile_info;
	std_picture_info.pQuantization = &quantization;
	std_picture_info.pSegmentation = nullptr;
	std_picture_info.pLoopFilter = nullptr;      // Skip loop filter for speed
	std_picture_info.pCDEF = nullptr;            // Skip CDEF for speed
	std_picture_info.pLoopRestoration = nullptr; // Skip loop restoration for speed
	std_picture_info.pGlobalMotion = nullptr;
	std_picture_info.pExtensionHeader = nullptr;
	std_picture_info.pBufferRemovalTimes = nullptr;

	picture_info = vk::VideoEncodeAV1PictureInfoKHR{};
	picture_info.predictionMode = is_key_frame ? vk::VideoEncodeAV1PredictionModeKHR::eIntraOnly : vk::VideoEncodeAV1PredictionModeKHR::eSingleReference;
	picture_info.rateControlGroup = is_key_frame ? vk::VideoEncodeAV1RateControlGroupKHR::eIntra : vk::VideoEncodeAV1RateControlGroupKHR::ePredictive;
	picture_info.constantQIndex = quantization.base_q_idx; // Explicitly set QP
	picture_info.pStdPictureInfo = &std_picture_info;

	// Update DPB reference info
	auto & i = dpb_std_info[slot];
	i.flags.disable_frame_end_update_cdf = 1; // Speed up
	i.flags.segmentation_enabled = 0;
	i.RefFrameId = frame_num_val;
	i.frame_type = std_picture_info.frame_type;
	i.OrderHint = 0; // No order hint used
	i.pExtensionHeader = nullptr;

	return &picture_info;
}

vk::ExtensionProperties wivrn::video_encoder_vulkan_av1::std_header_version()
{
	vk::ExtensionProperties std_header_version{};
	std_header_version.specVersion = VK_MAKE_VIDEO_STD_VERSION(1, 0, 0);
	strcpy(std_header_version.extensionName,
	       VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_EXTENSION_NAME);
	return std_header_version;
}