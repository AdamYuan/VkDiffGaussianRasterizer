//
// Created by adamyuan on 3/24/25.
//

#include "Rasterizer.hpp"

namespace vkgsraster {

myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardResetShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardReset.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardViewShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardView.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardDraw.vert.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardDraw.frag.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardCopy.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createBackwardResetShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/BackwardReset.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createBackwardViewShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/BackwardView.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createBackwardDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/BackwardDraw.vert.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createBackwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/BackwardDraw.frag.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> Rasterizer::createBackwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/BackwardCopy.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}

} // namespace vkgsraster