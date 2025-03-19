//
// Created by adamyuan on 3/17/25.
//

#pragma once

#include <myvk/BufferBase.hpp>
#include <myvk/ComputePipeline.hpp>
#include <myvk/DescriptorPool.hpp>
#include <myvk/DescriptorSet.hpp>
#include <array>

namespace VkGSRaster {

class DeviceSorter {
public:
	struct Args {
		myvk::Ptr<myvk::BufferBase> pKeyBuffer, pPayloadBuffer; // count * [uint]
	};

	struct Resource {
		myvk::Ptr<myvk::BufferBase> pTempKeyBuffer, pTempPayloadBuffer; // count * [uint]
		myvk::Ptr<myvk::BufferBase> pGlobalHistBuffer;
		myvk::Ptr<myvk::BufferBase> pPassHistBuffer;
		myvk::Ptr<myvk::BufferBase> pIndexBuffer;
		myvk::Ptr<myvk::BufferBase> pIndirectBuffer;

		void update(const myvk::Ptr<myvk::Device> &pDevice, uint32_t count, double growFactor = 1.5f);
	};

private:
	myvk::Ptr<myvk::ComputePipeline> mpResetPipeline, mpGlobalHistPipeline, mpScanHistPipeline, mpOneSweepPipeline;
	myvk::Ptr<myvk::DescriptorSet> mpResetDescriptorSet, mpGlobalHistDescriptorSet, mpScanHistDescriptorSet;
	std::array<myvk::Ptr<myvk::DescriptorSet>, >


};

}