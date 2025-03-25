//
// Created by adamyuan on 3/23/25.
//

#pragma once
#ifndef VKGSRASTER_RESOURCEUTIL_HPP
#define VKGSRASTER_RESOURCEUTIL_HPP

#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>

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

template <VkFormat Format_V>
inline void ResizeImage(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::ImageBase> &pImage,
                        VkImageUsageFlags imageUsage, uint32_t width, uint32_t height) {
	if (pImage == nullptr || pImage->GetExtent().width != width || pImage->GetExtent().height != height)
		pImage = myvk::Image::CreateTexture2D(pDevice, {width, height}, 1, Format_V, imageUsage);
}

} // namespace VkGSRaster

#endif
