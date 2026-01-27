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
#pragma once

#include "video_encoder_vulkan.h"

#include <memory>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace wivrn
{

class video_encoder_vulkan_av1 : public video_encoder_vulkan
{
	uint32_t frame_num = 0;
	VkVideoEncodeAV1StdFlagBitsKHR std_extension_profile = VK_VIDEO_ENCODE_AV1_STD_UNIFORM_TILE_SPACING_FLAG_SET_BIT_KHR;

	StdVideoAV1SequenceHeader sequence_header{};
	StdVideoAV1TileInfo tile_info{};
	StdVideoAV1Quantization quantization{};

	StdVideoEncodeAV1PictureInfo std_picture_info{};
	vk::VideoEncodeAV1PictureInfoKHR picture_info{};

	StdVideoEncodeAV1ReferenceInfo reference_info{};

	std::vector<StdVideoEncodeAV1ReferenceInfo> dpb_std_info;
	std::vector<vk::VideoEncodeAV1DpbSlotInfoKHR> dpb_std_slots;

	vk::VideoEncodeAV1RateControlInfoKHR rate_control_av1{};
	vk::VideoEncodeAV1RateControlLayerInfoKHR rate_control_layer_av1{};
	vk::VideoEncodeAV1RateControlInfoKHR rate_control_info{};

	video_encoder_vulkan_av1(wivrn_vk_bundle & vk,
	                         const vk::VideoCapabilitiesKHR & video_caps,
	                         const vk::VideoEncodeCapabilitiesKHR & encode_caps,
	                         uint8_t stream_idx,
	                         const encoder_settings & settings);

protected:
	std::vector<void *> setup_slot_info(size_t dpb_size) override;

	void * encode_info_next(uint32_t frame_num, size_t slot, std::optional<int32_t> ref_slot) override;
	vk::ExtensionProperties std_header_version() override;

	void send_idr_data() override;

public:
	static std::unique_ptr<video_encoder_vulkan_av1> create(wivrn_vk_bundle & vk,
	                                                        const encoder_settings & settings,
	                                                        uint8_t stream_idx);

	std::vector<uint8_t> get_sequence_headers();
};

} // namespace wivrn