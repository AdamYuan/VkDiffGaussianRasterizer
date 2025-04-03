//
// Created by adamyuan on 3/23/25.
//

#pragma once
#ifndef VKGSRASTER_RESOURCEUTIL_HPP
#define VKGSRASTER_RESOURCEUTIL_HPP

#include <myvk/Buffer.hpp>
#include <myvk/Image.hpp>

namespace vkgsraster {

template <VkDeviceSize ElementSize_V>
inline bool GrowBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount, double growFactor) {
	if (pBuffer == nullptr || pBuffer->GetSize() < elementCount * ElementSize_V || pBuffer->GetUsage() != bufferUsage) {
		VkDeviceSize allocSize = pBuffer ? VkDeviceSize(double(pBuffer->GetSize()) * growFactor) : 0;
		allocSize = std::max(allocSize, elementCount * ElementSize_V);
		pBuffer = myvk::Buffer::Create(pDevice, allocSize, 0, bufferUsage);
		return true;
	}
	return false;
}

template <VkDeviceSize ElementSize_V>
inline bool MakeBuffer(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::BufferBase> &pBuffer,
                       VkBufferUsageFlags bufferUsage, uint32_t elementCount) {
	if (pBuffer == nullptr || pBuffer->GetUsage() != bufferUsage) {
		pBuffer = myvk::Buffer::Create(pDevice, elementCount * ElementSize_V, 0, bufferUsage);
		return true;
	}
	return false;
}

template <VkFormat Format_V>
inline bool ResizeImage(const myvk::Ptr<myvk::Device> &pDevice, myvk::Ptr<myvk::ImageBase> &pImage,
                        VkImageUsageFlags imageUsage, uint32_t width, uint32_t height) {
	if (pImage == nullptr || pImage->GetExtent().width != width || pImage->GetExtent().height != height ||
	    pImage->GetUsage() != imageUsage) {
		pImage = myvk::Image::CreateTexture2D(pDevice, {width, height}, 1, Format_V, imageUsage);
		return true;
	}
	return false;
}

inline bool ResizeFramebuffer(const myvk::Ptr<myvk::RenderPass> &pRenderPass,
                              const std::vector<myvk::Ptr<myvk::ImageView>> &pImageViews,
                              myvk::Ptr<myvk::Framebuffer> &pFramebuffer, uint32_t width, uint32_t height) {
	if (pFramebuffer == nullptr || pFramebuffer->GetExtent().width != width ||
	    pFramebuffer->GetExtent().height != height || pFramebuffer->GetRenderPassPtr() != pRenderPass ||
	    pFramebuffer->GetImageViewPtrs() != pImageViews) {
		pFramebuffer = myvk::Framebuffer::Create(pRenderPass, pImageViews, {width, height});
		return true;
	}
	return false;
}

} // namespace vkgsraster

#endif
