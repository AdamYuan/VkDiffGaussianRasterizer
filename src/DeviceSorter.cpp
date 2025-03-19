//
// Created by adamyuan on 3/17/25.
//

#include "DeviceSorter.hpp"

#include <myvk/Buffer.hpp>
#include <shader/DeviceSorter/DeviceSorterSize.hpp>

namespace VkGSRaster {

void DeviceSorter::Resource::update(const myvk::Ptr<myvk::Device> &pDevice, uint32_t count, double growFactor) {
	const auto growBuffer = [&pDevice, growFactor](myvk::Ptr<myvk::BufferBase> &pBuffer, uint32_t targetUintCount,
	                                               VkBufferUsageFlags bufferUsage) {
		if (pBuffer == nullptr || pBuffer->GetSize() < targetUintCount * sizeof(uint32_t)) {
			VkDeviceSize allocSize = pBuffer ? VkDeviceSize(double(pBuffer->GetSize()) * growFactor) : 0;
			allocSize = std::max(allocSize, targetUintCount * sizeof(uint32_t));
			pBuffer = myvk::Buffer::Create(pDevice, allocSize, 0, bufferUsage);
		}
	};
	const auto makeBuffer = [&pDevice](myvk::Ptr<myvk::BufferBase> &pBuffer, uint32_t uintCount,
	                                   VkBufferUsageFlags bufferUsage) {
		if (pBuffer == nullptr)
			pBuffer = myvk::Buffer::Create(pDevice, uintCount * sizeof(uint32_t), 0, bufferUsage);
	};
	growBuffer(pTempKeyBuffer, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	growBuffer(pTempPayloadBuffer, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	growBuffer(pPassHistBuffer, PASS_COUNT * RADIX * ((count + SORT_PART_SIZE - 1) / SORT_PART_SIZE),
	           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pGlobalHistBuffer, PASS_COUNT * RADIX, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pIndexBuffer, PASS_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	makeBuffer(pIndirectBuffer, 6, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
}

} // namespace VkGSRaster