//
// Created by adamyuan on 3/24/25.
//

#include "RasterizerF32.hpp"

namespace vkgsraster {

myvk::Ptr<myvk::ShaderModule> RasterizerF32::createDrawVertShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/Rasterizer/Draw.vert.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createForwardResetShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/ForwardReset.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createForwardViewShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/ForwardView.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createForwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/ForwardDraw.geom.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createForwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/ForwardDraw.frag.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createForwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/ForwardCopy.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createBackwardResetShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/BackwardReset.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createBackwardViewShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/BackwardView.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createBackwardDrawGeomShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/BackwardDraw.geom.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createBackwardDrawFragShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/BackwardDraw.frag.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}
myvk::Ptr<myvk::ShaderModule> RasterizerF32::createBackwardCopyShader(const myvk::Ptr<myvk::Device> &pDevice) {
	static constexpr uint32_t kCode[] = {
#include <shader/RasterizerF32/BackwardCopy.comp.inl>

	};
	return myvk::ShaderModule::Create(pDevice, kCode, sizeof(kCode));
}

} // namespace vkgsraster