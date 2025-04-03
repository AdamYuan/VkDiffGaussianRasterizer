//
// Created by adamyuan on 3/24/25.
//

#include "Rasterizer.hpp"

namespace vkgsraster {

myvk::Ptr<myvk::ShaderModule> Rasterizer::createDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/Draw.vert.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
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
myvk::Ptr<myvk::ShaderModule> Rasterizer::createForwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/ForwardDraw.geom.inl>

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

} // namespace vkgsraster