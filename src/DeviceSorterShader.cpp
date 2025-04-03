//
// Created by adamyuan on 3/19/25.
//

#include "DeviceSorter.hpp"

namespace vkgsraster {

myvk::Ptr<myvk::ShaderModule> DeviceSorter::createResetShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/Reset.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}

myvk::Ptr<myvk::ShaderModule> DeviceSorter::createGlobalHistShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/GlobalHist.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}

myvk::Ptr<myvk::ShaderModule> DeviceSorter::createScanHistShader(const myvk::Ptr<myvk::Device> &pDevice) {
	auto subgroupSize = pDevice->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	if (subgroupSize == 32) {
		static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/ScanHist32.comp.inl>

		};
		return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	} else if (subgroupSize == 64) {
		static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/ScanHist64.comp.inl>

		};
		return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	}
	return nullptr;
}

myvk::Ptr<myvk::ShaderModule> DeviceSorter::createOneSweepShader(const myvk::Ptr<myvk::Device> &pDevice) {
	auto subgroupSize = pDevice->GetPhysicalDevicePtr()->GetProperties().vk11.subgroupSize;
	if (subgroupSize == 32) {
		static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/OneSweep32.comp.inl>

		};
		return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	} else if (subgroupSize == 64) {
		static constexpr uint32_t kCode[] = {
#include <shader/DeviceSorter/OneSweep64.comp.inl>

		};
		return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
	}
	return nullptr;
}

} // namespace vkgsraster
