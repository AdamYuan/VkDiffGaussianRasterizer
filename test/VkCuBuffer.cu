//
// Created by adamyuan on 4/1/25.
//

#include "VkCuBuffer.hpp"

#include <myvk/ExportBuffer.hpp>
#include <myvk/ExternalMemoryUtil.hpp>

myvk::Ptr<VkCuBuffer> VkCuBuffer::Create(const myvk::Ptr<myvk::Device> &pDevice, VkDeviceSize size,
                                         VkBufferUsageFlags usage, VkMemoryPropertyFlags memoryProperties,
                                         const std::vector<myvk::Ptr<myvk::Queue>> &pAccessQueues) {
	auto vkExtBufHandle = myvk::ExportBuffer::CreateHandle(pDevice, size, usage, memoryProperties, pAccessQueues);
	if (!bool(vkExtBufHandle))
		return nullptr;

	VkExternalMemoryHandleTypeFlagBits vkExtMemHandleType = myvk::GetExternalMemoryHandleType();
	cudaExternalMemoryHandleDesc cuExtMemHandleDesc = {
	    .size = size,
	};

	if (vkExtMemHandleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT)
		cuExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
	else if (vkExtMemHandleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT)
		cuExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
	else if (vkExtMemHandleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT)
		cuExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;

#ifdef _WIN64
	cuExtMemHandleDesc.handle.win32.handle = vkExtBufHandle.mem_handle;
#else
	cuExtMemHandleDesc.handle.fd = (int)(uintptr_t)vkExtBufHandle.mem_handle;
#endif

	float *pCuMapped;
	cudaExternalMemory_t cuExtMem;
	if (cudaImportExternalMemory(&cuExtMem, &cuExtMemHandleDesc) != cudaSuccess) {
		myvk::ExportBuffer::DestroyHandle(pDevice, vkExtBufHandle);
		return nullptr;
	}
	cudaExternalMemoryBufferDesc cuExtMemBufferDesc = {};
	cuExtMemBufferDesc.offset = 0;
	cuExtMemBufferDesc.size = size;
	cuExtMemBufferDesc.flags = 0;
	if (cudaExternalMemoryGetMappedBuffer((void **)&pCuMapped, cuExtMem, &cuExtMemBufferDesc) != cudaSuccess) {
		cudaDestroyExternalMemory(cuExtMem);
		myvk::ExportBuffer::DestroyHandle(pDevice, vkExtBufHandle);
		return nullptr;
	}

	auto pRet = myvk::MakePtr<VkCuBuffer>();
	pRet->mpDevice = pDevice;

	pRet->m_buffer = vkExtBufHandle.buffer;
	pRet->m_usage = usage;
	pRet->m_size = size;

	pRet->mDeviceMemory = vkExtBufHandle.device_memory;
	pRet->mpCudaMapped = pCuMapped;
	pRet->mCudaExtMemory = (uintptr_t)cuExtMem;

	return pRet;
}

VkCuBuffer::~VkCuBuffer() {
	if (mCudaExtMemory)
		cudaDestroyExternalMemory((cudaExternalMemory_t)mCudaExtMemory);
	myvk::ExportBuffer::DestroyHandle(mpDevice, {.buffer = m_buffer, .device_memory = mDeviceMemory});
}