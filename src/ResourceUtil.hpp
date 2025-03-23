//
// Created by adamyuan on 3/23/25.
//

#pragma once
#ifndef VKGSRASTER_RESOURCEUTIL_HPP
#define VKGSRASTER_RESOURCEUTIL_HPP

#include <myvk/Buffer.hpp>

namespace VkGSRaster {

template <VkDeviceSize ElementSize_V>
inline void GrowBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount, double growFactor) {
	if (pBuffer == nullptr || pBuffer->GetSize() < elementCount * ElementSize_V) {
		VkDeviceSize allocSize = pBuffer ? VkDeviceSize(double(pBuffer->GetSize()) * growFactor) : 0;
		allocSize = std::max(allocSize, elementCount * ElementSize_V);
		pBuffer = myvk::Buffer::Create(pDevice, allocSize, 0, bufferUsage);
	}
}

template <VkDeviceSize ElementSize_V>
inline void MakeBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount) {
	if (pBuffer == nullptr)
		pBuffer = myvk::Buffer::Create(pDevice, elementCount * ElementSize_V, 0, bufferUsage);
}

template <typename T>
inline void CmdInitializeBuffer(const myvk::Ptr<myvk::CommandBuffer> &pCommandBuffer,
                                const myvk::Ptr<myvk::BufferBase> &pBuffer, const T &data) {
	vkCmdUpdateBuffer(pCommandBuffer->GetHandle(), pBuffer->GetHandle(), 0, sizeof(T), &data);
	pCommandBuffer->CmdPipelineBarrier2(
	    {},
	    {pBuffer->GetMemoryBarrier2(
	        {VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_TRANSFER_WRITE_BIT},
	        {VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT})},
	    {});
}

} // namespace VkGSRaster

#endif
