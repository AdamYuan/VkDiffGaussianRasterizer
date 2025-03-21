//
// Created by adamyuan on 3/21/25.
//

#include "Context.hpp"

namespace VkGSRaster {

namespace {
const auto createDeviceQueue =
    [](auto &&physicalDeviceSelectFunc) -> std::tuple<myvk::Ptr<myvk::Device>, myvk::Ptr<myvk::Queue>> {
	auto pInstance = myvk::Instance::Create({});
	auto pPhysicalDevices = myvk::PhysicalDevice::Fetch(pInstance);
	myvk::Ptr<myvk::PhysicalDevice> pPhysicalDevice = physicalDeviceSelectFunc(pPhysicalDevices);
	auto features = pPhysicalDevice->GetDefaultFeatures();
	features.vk13.synchronization2 = VK_TRUE;
	features.vk13.computeFullSubgroups = VK_TRUE;
	myvk::Ptr<myvk::Queue> pQueue;
	auto pDevice =
	    myvk::Device::Create(pPhysicalDevice, myvk::GenericQueueSelector{&pQueue}, features,
	                         {VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
	return {pDevice, pQueue};
};
} // namespace

Context::Context(uint32_t physicalDeviceID) {
	std::tie(mpDevice, mpQueue) = createDeviceQueue(
	    [&](const std::vector<myvk::Ptr<myvk::PhysicalDevice>> &pPhysicalDevices) -> myvk::Ptr<myvk::PhysicalDevice> {
		    if (pPhysicalDevices.empty())
			    return nullptr;
		    if (physicalDeviceID < pPhysicalDevices.size())
			    return pPhysicalDevices[physicalDeviceID];
		    // Or find an optimal physicalDevice
		    for (const auto &pCandidate : pPhysicalDevices) {
			    if (pCandidate->GetExtensionSupport(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME) &&
			        pCandidate->GetProperties().vk10.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				    return pCandidate;
			    }
		    }
		    return pPhysicalDevices[0];
	    });
}

} // namespace VkGSRaster