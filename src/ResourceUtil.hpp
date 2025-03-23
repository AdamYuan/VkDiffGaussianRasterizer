//
// Created by adamyuan on 3/23/25.
//

#pragma once
#ifndef VKGSRASTER_RESOURCEUTIL_HPP
#define VKGSRASTER_RESOURCEUTIL_HPP

#include <myvk/Buffer.hpp>

namespace VkGSRaster {

template <VkDeviceSize ElementSize_V>
inline void growBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount, double growFactor) {
	if (pBuffer == nullptr || pBuffer->GetSize() < elementCount * ElementSize_V) {
		VkDeviceSize allocSize = pBuffer ? VkDeviceSize(double(pBuffer->GetSize()) * growFactor) : 0;
		allocSize = std::max(allocSize, elementCount * ElementSize_V);
		pBuffer = myvk::Buffer::Create(pDevice, allocSize, 0, bufferUsage);
	}
}

template <VkDeviceSize ElementSize_V>
inline void makeBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount) {
	if (pBuffer == nullptr)
		pBuffer = myvk::Buffer::Create(pDevice, elementCount * ElementSize_V, 0, bufferUsage);
}

} // namespace VkGSRaster

#endif
