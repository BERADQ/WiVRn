/*
 * WiVRn VR streaming
 * Copyright (C) 2024  Patrick Nicolas <patricknicolas@laposte.net>
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

#include <algorithm>
#include <cstring>
#include <limits>

namespace
{
uint32_t align_div(uint32_t value, uint32_t divisor)
{
	return (value + divisor - 1) / divisor;
}

uint8_t bits_for(uint32_t value)
{
	uint8_t bits = 0;
	while (value > 0)
	{
		value >>= 1;
		++bits;
	}
	return std::max<uint8_t>(bits, 1);
}
} // namespace

wivrn::video_encoder_vulkan_av1::video_encoder_vulkan_av1(
        wivrn_vk_bundle & vk,
        const vk::VideoCapabilitiesKHR & video_caps,
        const vk::VideoEncodeCapabilitiesKHR & encode_caps,
        const vk::VideoEncodeAV1CapabilitiesKHR & encode_av1_caps,
        uint8_t stream_idx,
        const encoder_settings & settings) :
        video_encoder_vulkan(vk, video_caps, encode_caps, stream_idx, settings)
{
	if (not std::any_of(vk.device_extensions.begin(), vk.device_extensions.end(), [](const char * ext) { return std::strcmp(ext, VK_KHR_VIDEO_ENCODE_AV1_EXTENSION_NAME) == 0; }))
	{
		throw std::runtime_error("Vulkan video encode AV1 extension not available");
	}

	if (settings.bit_depth != 8 && settings.bit_depth != 10)
		throw std::runtime_error("av1 encoder supports 8-bit or 10-bit only");

	configure_from_caps(encode_av1_caps);

	color_config = {
	        .flags = {
	                .mono_chrome = 0,
	                .color_range = 1,
	                .separate_uv_delta_q = 0,
	                .color_description_present_flag = 1,
	        },
	        .BitDepth = static_cast<uint8_t>(settings.bit_depth),
	        .subsampling_x = 1,
	        .subsampling_y = 1,
	        .reserved1 = 0,
	        .color_primaries = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709,
	        .transfer_characteristics = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_709,
	        .matrix_coefficients = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_709,
	        .chroma_sample_position = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_COLOCATED,
	};

	seq_header = {
	        .flags = {
	                .still_picture = 0,
	                .reduced_still_picture_header = 0,
	                .use_128x128_superblock = superblock_size == 128 ? 1u : 0u,
	                .enable_filter_intra = 0,
	                .enable_intra_edge_filter = 0,
	                .enable_interintra_compound = 0,
	                .enable_masked_compound = 0,
	                .enable_warped_motion = 0,
	                .enable_dual_filter = 0,
	                .enable_order_hint = 1,
	                .enable_jnt_comp = 0,
	                .enable_ref_frame_mvs = 0,
	                .frame_id_numbers_present_flag = 0,
	                .enable_superres = 0,
	                .enable_cdef = 0,
	                .enable_restoration = 0,
	                .film_grain_params_present = 0,
	                .timing_info_present_flag = 0,
	                .initial_display_delay_present_flag = 0,
	        },
	        .seq_profile = STD_VIDEO_AV1_PROFILE_MAIN,
	        .frame_width_bits_minus_1 = static_cast<uint8_t>(bits_for(aligned_extent.width - 1) - 1),
	        .frame_height_bits_minus_1 = static_cast<uint8_t>(bits_for(aligned_extent.height - 1) - 1),
	        .max_frame_width_minus_1 = static_cast<uint16_t>(aligned_extent.width - 1),
	        .max_frame_height_minus_1 = static_cast<uint16_t>(aligned_extent.height - 1),
	        .delta_frame_id_length_minus_2 = 0,
	        .additional_frame_id_length_minus_1 = 0,
	        .order_hint_bits_minus_1 = static_cast<uint8_t>(order_hint_bits - 1),
	        .seq_force_integer_mv = 0,
	        .seq_force_screen_content_tools = STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS,
	        .reserved1 = {},
	        .pColorConfig = &color_config,
	        .pTimingInfo = nullptr,
	};

	const uint32_t sb_cols = align_div(aligned_extent.width, superblock_size);
	const uint32_t sb_rows = align_div(aligned_extent.height, superblock_size);
	const uint32_t mi_cols = align_div(aligned_extent.width, 4);
	const uint32_t mi_rows = align_div(aligned_extent.height, 4);

	mi_col_starts = {0u, static_cast<uint16_t>(mi_cols)};
	mi_row_starts = {0u, static_cast<uint16_t>(mi_rows)};
	if (!tile_widths_sb.empty())
		tile_widths_sb[0] = static_cast<uint16_t>(sb_cols - 1);
	if (!tile_heights_sb.empty())
		tile_heights_sb[0] = static_cast<uint16_t>(sb_rows - 1);

	tile_info = {
	        .flags = {
	                .uniform_tile_spacing_flag = (std_flags & vk::VideoEncodeAV1StdFlagBitsKHR::eUniformTileSpacingFlagSet) ? 1u : 0u,
	        },
	        .TileCols = 1,
	        .TileRows = 1,
	        .context_update_tile_id = 0,
	        .tile_size_bytes_minus_1 = 0,
	        .reserved1 = {},
	        .pMiColStarts = mi_col_starts.data(),
	        .pMiRowStarts = mi_row_starts.data(),
	        .pWidthInSbsMinus1 = tile_widths_sb.data(),
	        .pHeightInSbsMinus1 = tile_heights_sb.data(),
	};

	quantization = {
	        .flags = {
	                .using_qmatrix = 0,
	                .diff_uv_delta = 0,
	        },
	        .base_q_idx = 128,
	        .DeltaQYDc = 0,
	        .DeltaQUDc = 0,
	        .DeltaQUAc = 0,
	        .DeltaQVDc = 0,
	        .DeltaQVAc = 0,
	        .qm_y = 0,
	        .qm_u = 0,
	        .qm_v = 0,
	};

	loop_filter = {
	        .flags = {
	                .loop_filter_delta_enabled = 0,
	                .loop_filter_delta_update = 0,
	        },
	        .loop_filter_level = {},
	        .loop_filter_sharpness = 0,
	        .update_ref_delta = 0,
	        .loop_filter_ref_deltas = {},
	        .update_mode_delta = 0,
	        .loop_filter_mode_deltas = {},
	};

	cdef = {
	        .cdef_damping_minus_3 = 0,
	        .cdef_bits = 0,
	        .cdef_y_pri_strength = {},
	        .cdef_y_sec_strength = {},
	        .cdef_uv_pri_strength = {},
	        .cdef_uv_sec_strength = {},
	};

	loop_restoration = {
	        .FrameRestorationType = {STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
	                                 STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
	                                 STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE},
	        .LoopRestorationSize = {0, 0, 0},
	};

	global_motion = {
	        .GmType = {},
	        .gm_params = {},
	};

	segmentation = {};

	rate_control_layer.pNext = &rate_control_layer_av1;
	if (rate_control)
		rate_control->pNext = &rate_control_av1;
}

void wivrn::video_encoder_vulkan_av1::configure_from_caps(const vk::VideoEncodeAV1CapabilitiesKHR & encode_av1_caps)
{
	std_flags = encode_av1_caps.stdSyntaxFlags;
	if (encode_av1_caps.superblockSizes & vk::VideoEncodeAV1SuperblockSizeFlagBitsKHR::e64)
		superblock_size = 64;
	else
		superblock_size = 128;

	rate_control_av1 = vk::VideoEncodeAV1RateControlInfoKHR{
	        .flags = vk::VideoEncodeAV1RateControlFlagBitsKHR::eRegularGop,
	        .gopFrameCount = std::numeric_limits<uint32_t>::max(),
	        .keyFramePeriod = std::numeric_limits<uint32_t>::max(),
	        .consecutiveBipredictiveFrameCount = 0,
	        .temporalLayerCount = 1,
	};

	rate_control_layer_av1 = vk::VideoEncodeAV1RateControlLayerInfoKHR{
	        .useMinQIndex = VK_FALSE,
	        .minQIndex = {
	                .intraQIndex = encode_av1_caps.minQIndex,
	                .predictiveQIndex = encode_av1_caps.minQIndex,
	                .bipredictiveQIndex = encode_av1_caps.minQIndex,
	        },
	        .useMaxQIndex = VK_FALSE,
	        .maxQIndex = {
	                .intraQIndex = encode_av1_caps.maxQIndex,
	                .predictiveQIndex = encode_av1_caps.maxQIndex,
	                .bipredictiveQIndex = encode_av1_caps.maxQIndex,
	        },
	        .useMaxFrameSize = VK_FALSE,
	        .maxFrameSize = {},
	};
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
		throw std::runtime_error("av1 encoder supports 8-bit or 10-bit only");

	vk::StructureChain video_profile_info{
	        vk::VideoProfileInfoKHR{
	                .videoCodecOperation = vk::VideoCodecOperationFlagBitsKHR::eEncodeAv1,
	                .chromaSubsampling = vk::VideoChromaSubsamplingFlagBitsKHR::e420,
	                .lumaBitDepth = bit_depth == 10 ? vk::VideoComponentBitDepthFlagBitsKHR::e10 : vk::VideoComponentBitDepthFlagBitsKHR::e8,
	                .chromaBitDepth = bit_depth == 10 ? vk::VideoComponentBitDepthFlagBitsKHR::e10 : vk::VideoComponentBitDepthFlagBitsKHR::e8,
	        },
	        vk::VideoEncodeAV1ProfileInfoKHR{
	                .stdProfile = STD_VIDEO_AV1_PROFILE_MAIN,
	        },
	        vk::VideoEncodeUsageInfoKHR{
	                .videoUsageHints = vk::VideoEncodeUsageFlagBitsKHR::eStreaming,
	                .videoContentHints = vk::VideoEncodeContentFlagBitsKHR::eRendered,
	                .tuningMode = vk::VideoEncodeTuningModeKHR::eUltraLowLatency,
	        }};

	try
	{
		auto [video_caps, encode_caps, encode_av1_caps] =
		        phys_dev.getVideoCapabilitiesKHR<
		                vk::VideoCapabilitiesKHR,
		                vk::VideoEncodeCapabilitiesKHR,
		                vk::VideoEncodeAV1CapabilitiesKHR>(video_profile_info.get());

		video_caps.maxDpbSlots = std::min(video_caps.maxDpbSlots, uint32_t(STD_VIDEO_AV1_NUM_REF_FRAMES));
		return std::make_tuple(video_caps, encode_caps, encode_av1_caps, video_profile_info);
	}
	catch (...)
	{}

	video_profile_info.unlink<vk::VideoEncodeUsageInfoKHR>();
	auto [video_caps, encode_caps, encode_av1_caps] =
	        phys_dev.getVideoCapabilitiesKHR<
	                vk::VideoCapabilitiesKHR,
	                vk::VideoEncodeCapabilitiesKHR,
	                vk::VideoEncodeAV1CapabilitiesKHR>(video_profile_info.get());
	video_caps.maxDpbSlots = std::min(video_caps.maxDpbSlots, uint32_t(STD_VIDEO_AV1_NUM_REF_FRAMES));
	return std::make_tuple(video_caps, encode_caps, encode_av1_caps, video_profile_info);
}

std::unique_ptr<wivrn::video_encoder_vulkan_av1> wivrn::video_encoder_vulkan_av1::create(
        wivrn_vk_bundle & vk,
        const encoder_settings & settings,
        uint8_t stream_idx)
{
	auto [video_caps, encode_caps, encode_av1_caps, video_profile_info] = get_video_caps(vk.physical_device, settings.bit_depth);

	std::unique_ptr<video_encoder_vulkan_av1> self(
	        new video_encoder_vulkan_av1(vk, video_caps, encode_caps, encode_av1_caps, stream_idx, settings));

	vk::VideoEncodeAV1SessionParametersCreateInfoKHR session_params_info{
	        .pStdSequenceHeader = &self->seq_header,
	        .pStdDecoderModelInfo = nullptr,
	        .stdOperatingPointCount = 0,
	        .pStdOperatingPoints = nullptr,
	};

	vk::VideoEncodeAV1SessionCreateInfoKHR session_create_info{
	        .useMaxLevel = false,
	        .maxLevel = encode_av1_caps.maxLevel,
	};

	if (encode_av1_caps.requiresGopRemainingFrames)
	{
		self->gop_info = vk::VideoEncodeAV1GopRemainingFrameInfoKHR{
		        .useGopRemainingFrames = true,
		        .gopRemainingIntra = 0,
		        .gopRemainingPredictive = std::numeric_limits<uint32_t>::max(),
		        .gopRemainingBipredictive = 0,
		};
		self->rate_control_av1.pNext = &self->gop_info;
	}

	self->init(video_caps, video_profile_info.get(), &session_create_info, &session_params_info);

	return self;
}

void wivrn::video_encoder_vulkan_av1::send_idr_data()
{
	// AV1 sequence headers are emitted by the encoder as needed.
}

void * wivrn::video_encoder_vulkan_av1::encode_info_next(uint32_t frame_num, size_t slot, std::optional<int32_t> ref_slot)
{
	const bool is_keyframe = !ref_slot;

	std::fill(reference_name_slot_indices.begin(), reference_name_slot_indices.end(), -1);
	if (ref_slot)
		reference_name_slot_indices[0] = *ref_slot;

	std_picture_info = {};
	std_picture_info.flags.error_resilient_mode = 0;
	std_picture_info.flags.disable_cdf_update = 0;
	std_picture_info.flags.use_superres = 0;
	std_picture_info.flags.render_and_frame_size_different = (aligned_extent.width != extent.width || aligned_extent.height != extent.height);
	std_picture_info.flags.allow_screen_content_tools = 1;
	std_picture_info.flags.is_filter_switchable = 0;
	std_picture_info.flags.force_integer_mv = 0;
	std_picture_info.flags.frame_size_override_flag = 0;
	std_picture_info.flags.buffer_removal_time_present_flag = 0;
	std_picture_info.flags.allow_intrabc = 0;
	std_picture_info.flags.frame_refs_short_signaling = 0;
	std_picture_info.flags.allow_high_precision_mv = 1;
	std_picture_info.flags.is_motion_mode_switchable = 0;
	std_picture_info.flags.use_ref_frame_mvs = 0;
	std_picture_info.flags.disable_frame_end_update_cdf = 0;
	std_picture_info.flags.allow_warped_motion = 0;
	std_picture_info.flags.reduced_tx_set = 0;
	std_picture_info.flags.skip_mode_present = 0;
	std_picture_info.flags.delta_q_present = 0;
	std_picture_info.flags.delta_lf_present = 0;
	std_picture_info.flags.delta_lf_multi = 0;
	std_picture_info.flags.segmentation_enabled = 0;
	std_picture_info.flags.segmentation_update_map = 0;
	std_picture_info.flags.segmentation_temporal_update = 0;
	std_picture_info.flags.segmentation_update_data = 0;
	std_picture_info.flags.UsesLr = 0;
	std_picture_info.flags.usesChromaLr = 0;
	std_picture_info.flags.show_frame = 1;
	std_picture_info.flags.showable_frame = 1;
	std_picture_info.frame_type = is_keyframe ? STD_VIDEO_AV1_FRAME_TYPE_KEY : STD_VIDEO_AV1_FRAME_TYPE_INTER;
	std_picture_info.frame_presentation_time = frame_num;
	std_picture_info.current_frame_id = frame_num;
	std_picture_info.order_hint = static_cast<uint8_t>(frame_num & ((1u << order_hint_bits) - 1));
	std_picture_info.primary_ref_frame = ref_slot ? 0 : STD_VIDEO_AV1_PRIMARY_REF_NONE;
	std_picture_info.refresh_frame_flags = static_cast<uint8_t>(is_keyframe ? 0xFF : 0x01);
	std_picture_info.coded_denom = 0;
	std_picture_info.render_width_minus_1 = static_cast<uint16_t>(extent.width - 1);
	std_picture_info.render_height_minus_1 = static_cast<uint16_t>(extent.height - 1);
	std_picture_info.interpolation_filter = STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE;
	std_picture_info.TxMode = STD_VIDEO_AV1_TX_MODE_SELECT;
	std_picture_info.delta_q_res = 0;
	std_picture_info.delta_lf_res = 0;
	std::fill(std_picture_info.ref_order_hint, std_picture_info.ref_order_hint + STD_VIDEO_AV1_NUM_REF_FRAMES, 0);
	std::fill(std_picture_info.ref_frame_idx, std_picture_info.ref_frame_idx + STD_VIDEO_AV1_REFS_PER_FRAME, static_cast<int8_t>(-1));
	std::fill(std_picture_info.delta_frame_id_minus_1, std_picture_info.delta_frame_id_minus_1 + STD_VIDEO_AV1_REFS_PER_FRAME, 0);
	std_picture_info.pTileInfo = &tile_info;
	std_picture_info.pQuantization = &quantization;
	std_picture_info.pSegmentation = &segmentation;
	std_picture_info.pLoopFilter = &loop_filter;
	std_picture_info.pCDEF = &cdef;
	std_picture_info.pLoopRestoration = &loop_restoration;
	std_picture_info.pGlobalMotion = &global_motion;
	std_picture_info.pExtensionHeader = nullptr;
	std_picture_info.pBufferRemovalTimes = nullptr;

	if (ref_slot)
		std_picture_info.ref_frame_idx[0] = 0;

	if (ref_slot)
	{
		const auto & ref_info = dpb_std_info[*ref_slot];
		std_picture_info.ref_order_hint[0] = ref_info.OrderHint;
	}

	picture_info = vk::VideoEncodeAV1PictureInfoKHR{};
	picture_info.predictionMode = is_keyframe ? vk::VideoEncodeAV1PredictionModeKHR::eIntraOnly
	                                          : vk::VideoEncodeAV1PredictionModeKHR::eSingleReference;
	picture_info.rateControlGroup = is_keyframe ? vk::VideoEncodeAV1RateControlGroupKHR::eIntra
	                                            : vk::VideoEncodeAV1RateControlGroupKHR::ePredictive;
	picture_info.constantQIndex = 0;
	picture_info.pStdPictureInfo = &std_picture_info;
	picture_info.referenceNameSlotIndices = reference_name_slot_indices;
	picture_info.primaryReferenceCdfOnly = VK_FALSE;
	picture_info.generateObuExtensionHeader = VK_FALSE;

	auto & i = dpb_std_info[slot];
	i = {};
	i.flags.disable_frame_end_update_cdf = 0;
	i.flags.segmentation_enabled = 0;
	i.RefFrameId = frame_num;
	i.frame_type = std_picture_info.frame_type;
	i.OrderHint = std_picture_info.order_hint;
	i.pExtensionHeader = nullptr;

	return &picture_info;
}

vk::ExtensionProperties wivrn::video_encoder_vulkan_av1::std_header_version()
{
	vk::ExtensionProperties std_header_version{
	        .specVersion = VK_MAKE_VIDEO_STD_VERSION(1, 0, 0),
	};
	strcpy(std_header_version.extensionName,
	       VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_EXTENSION_NAME);
	return std_header_version;
}
