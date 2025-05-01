//
// Created by adamyuan on 4/1/25.
//

#include "CuTileRasterizer.hpp"

#include "GSModel.hpp"
#include "VkCuBuffer.hpp"
#include <rasterizer.h>

#define cudaCheckError() \
	{ \
		cudaError_t e = cudaGetLastError(); \
		if (e != cudaSuccess) { \
			printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(0); \
		} \
	}

char *CuTileRasterizer::Resource::ResizeableBuffer::Update(std::size_t updateSize) {
	if (updateSize > size) {
		if (data)
			cudaFree(data);
		cudaMalloc(&data, updateSize);
		printf("CuBuffer::Update %zu -> %zu\n", size, updateSize);
		size = updateSize;
		cudaCheckError();
	}
	return data;
}

void CuTileRasterizer::CameraArgs::Update(const vkgsraster::Rasterizer::CameraArgs &vkCamera) {
	width = (int)vkCamera.width;
	height = (int)vkCamera.height;
	tanFovX = vkgsraster::Rasterizer::CameraArgs::GetTanFovFromFocal(vkCamera.focalX, vkCamera.width);
	tanFovY = vkgsraster::Rasterizer::CameraArgs::GetTanFovFromFocal(vkCamera.focalY, vkCamera.height);

	using Mat4 = std::array<float, 16>;
	using Mat3 = std::array<float, 9>;
	const auto $4 = [](Mat4 &mat, uint32_t i, uint32_t j) -> float & { return mat[i + j * 4]; };
	const auto $4c = [](const Mat4 &mat, uint32_t i, uint32_t j) -> float { return mat[i + j * 4]; };
	const auto $3c = [](const Mat3 &mat, uint32_t i, uint32_t j) -> float { return mat[i + j * 3]; };
	const auto mul4 = [&](const Mat4 &l, const Mat4 &r) -> Mat4 {
		Mat4 m{};
		for (uint32_t k = 0; k < 4; ++k)
			for (uint32_t j = 0; j < 4; ++j)
				for (uint32_t i = 0; i < 4; ++i) {
					$4(m, i, j) += $4c(l, i, k) * $4c(r, k, j);
				}
		return m;
	};

	Mat4 camViewMat{};
	{
		for (uint32_t j = 0; j < 3; ++j)
			for (uint32_t i = 0; i < 3; ++i)
				$4(camViewMat, i, j) = $3c(vkCamera.viewMat, i, j);
		for (uint32_t i = 0; i < 3; ++i)
			$4(camViewMat, i, 3) = -(vkCamera.pos[0] * $4c(camViewMat, i, 0)   //
			                         + vkCamera.pos[1] * $4c(camViewMat, i, 1) //
			                         + vkCamera.pos[2] * $4c(camViewMat, i, 2));
		$4(camViewMat, 3, 3) = 1.0f;
	}

	Mat4 camProjMat{};
	{
		float zNear = 0.01f;
		float zFar = 100.0f;
		float top = tanFovY * zNear;
		float bottom = -top;
		float right = tanFovX * zNear;
		float left = -right;
		float zSign = 1.0f;
		$4(camProjMat, 0, 0) = 2.0f * zNear / (right - left);
		$4(camProjMat, 1, 1) = 2.0f * zNear / (top - bottom);
		$4(camProjMat, 0, 2) = (right + left) / (right - left);
		$4(camProjMat, 1, 2) = (top + bottom) / (top - bottom);
		$4(camProjMat, 3, 2) = zSign;
		$4(camProjMat, 2, 2) = zSign * zFar / (zFar - zNear);
		$4(camProjMat, 2, 3) = -(zFar * zNear) / (zFar - zNear);

		camProjMat = mul4(camProjMat, camViewMat);
	}

	if (projMat)
		cudaFree((void *)projMat);
	float *deviceCamProjMat;
	cudaMalloc(&deviceCamProjMat, sizeof(Mat4));
	cudaMemcpy(deviceCamProjMat, camProjMat.data(), sizeof(Mat4), cudaMemcpyHostToDevice);
	projMat = deviceCamProjMat;
	cudaCheckError();

	if (viewMat)
		cudaFree((void *)viewMat);
	float *deviceCamViewMat;
	cudaMalloc(&deviceCamViewMat, sizeof(Mat4));
	cudaMemcpy(deviceCamViewMat, camViewMat.data(), sizeof(Mat4), cudaMemcpyHostToDevice);
	viewMat = deviceCamViewMat;
	cudaCheckError();

	if (pos)
		cudaFree((void *)pos);
	float *deviceCamPos;
	cudaMalloc(&deviceCamPos, sizeof(vkCamera.pos));
	cudaMemcpy(deviceCamPos, vkCamera.pos.data(), sizeof(vkCamera.pos), cudaMemcpyHostToDevice);
	pos = deviceCamPos;
	cudaCheckError();
}

void CuTileRasterizer::FwdROArgs::Update(const vkgsraster::Rasterizer::FwdROArgs &vkROArgs) {
	camera.Update(vkROArgs.camera);
	splatCount = vkROArgs.splatCount;

	splats = {
	    .means = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pMeanBuffer)->GetCudaMappedPtr<float>(),
	    .scales = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pScaleBuffer)->GetCudaMappedPtr<float>(),
	    .rotates = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pRotateBuffer)->GetCudaMappedPtr<float>(),
	    .opacities = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pOpacityBuffer)->GetCudaMappedPtr<float>(),
	    .shs = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pSHBuffer)->GetCudaMappedPtr<float>(),
	};

	if (bgColor)
		cudaFree((void *)bgColor);
	float *deviceBgColor;
	cudaMalloc(&deviceBgColor, sizeof(vkROArgs.bgColor));
	cudaMemcpy(deviceBgColor, vkROArgs.bgColor.data(), sizeof(vkROArgs.bgColor), cudaMemcpyHostToDevice);
	bgColor = deviceBgColor;
	cudaCheckError();
}

void CuTileRasterizer::FwdRWArgs::Update(const vkgsraster::Rasterizer::FwdRWArgs &vkRWArgs) {
	outPixels = std::static_pointer_cast<VkCuBuffer>(vkRWArgs.pOutPixelBuffer)->GetCudaMappedPtr<float>();
}

void CuTileRasterizer::BwdROArgs::Update(const FwdROArgs &fwdROArgs,
                                         const vkgsraster::Rasterizer::BwdROArgs &vkROArgs) {
	fwd = fwdROArgs;
	dL_dPixels = std::static_pointer_cast<VkCuBuffer>(vkROArgs.pdL_dPixelBuffer)->GetCudaMappedPtr<float>();
}

void CuTileRasterizer::BwdRWArgs::Update(const vkgsraster::Rasterizer::BwdRWArgs &vkRWArgs) {
	dL_dSplats = {
	    .means = std::static_pointer_cast<VkCuBuffer>(vkRWArgs.dL_dSplats.pMeanBuffer)->GetCudaMappedPtr<float>(),
	    .scales = std::static_pointer_cast<VkCuBuffer>(vkRWArgs.dL_dSplats.pScaleBuffer)->GetCudaMappedPtr<float>(),
	    .rotates = std::static_pointer_cast<VkCuBuffer>(vkRWArgs.dL_dSplats.pRotateBuffer)->GetCudaMappedPtr<float>(),
	    .opacities =
	        std::static_pointer_cast<VkCuBuffer>(vkRWArgs.dL_dSplats.pOpacityBuffer)->GetCudaMappedPtr<float>(),
	    .shs = std::static_pointer_cast<VkCuBuffer>(vkRWArgs.dL_dSplats.pSHBuffer)->GetCudaMappedPtr<float>(),
	};
}

void CuTileRasterizer::Forward(const FwdROArgs &roArgs, const FwdRWArgs &rwArgs, Resource &resource,
                               const PerfQuery &perfQuery) {

	cudaDeviceSynchronize();

	resource.numRendered = CudaRasterizer::Rasterizer::forward(
	    [&](std::size_t size) { return resource.geometryBuffer.Update(size); },
	    [&](std::size_t size) { return resource.binningBuffer.Update(size); },
	    [&](std::size_t size) { return resource.imageBuffer.Update(size); }, (int)roArgs.splatCount, GSModel::kSHDegree,
	    GSModel::kSHSize, roArgs.bgColor, roArgs.camera.width, roArgs.camera.height, roArgs.splats.means,
	    roArgs.splats.shs, nullptr, roArgs.splats.opacities, roArgs.splats.scales, 1.0f, roArgs.splats.rotates, nullptr,
	    roArgs.camera.viewMat, roArgs.camera.projMat, roArgs.camera.pos, roArgs.camera.tanFovX, roArgs.camera.tanFovY,
	    false, rwArgs.outPixels, nullptr, false, perfQuery);

	cudaDeviceSynchronize();
}

void CuTileRasterizer::Backward(const BwdROArgs &roArgs, const BwdRWArgs &rwArgs, Resource &resource,
                                const PerfQuery &perfQuery) {
	/* torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
	torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
	torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
	torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options()); */

	uint32_t dL_dmean2D_count = roArgs.fwd.splatCount * 3;
	uint32_t dL_dcolor_count = roArgs.fwd.splatCount * 3;
	uint32_t dL_dconic_count = roArgs.fwd.splatCount * 4;
	uint32_t dL_dcov3D_count = roArgs.fwd.splatCount * 6;
	uint32_t dL_count = dL_dmean2D_count + dL_dcolor_count + dL_dconic_count + dL_dcov3D_count;

	auto dL = reinterpret_cast<float *>(resource.dLBuffer.Update(dL_count * sizeof(float)));

	cudaDeviceSynchronize();

	cudaMemset(dL, 0, dL_count * sizeof(float));

	float *dL_dmean2Ds = dL;
	float *dL_dcolors = dL_dmean2Ds + dL_dmean2D_count;
	float *dL_dconics = dL_dcolors + dL_dcolor_count;
	float *dL_dcov3D = dL_dconics + dL_dconic_count;

	CudaRasterizer::Rasterizer::backward(
	    (int)roArgs.fwd.splatCount, GSModel::kSHDegree, GSModel::kSHSize, resource.numRendered, roArgs.fwd.bgColor,
	    roArgs.fwd.camera.width, roArgs.fwd.camera.height, roArgs.fwd.splats.means, roArgs.fwd.splats.shs, nullptr,
	    roArgs.fwd.splats.scales, 1.0f, roArgs.fwd.splats.rotates, nullptr, roArgs.fwd.camera.viewMat,
	    roArgs.fwd.camera.projMat, roArgs.fwd.camera.pos, roArgs.fwd.camera.tanFovX, roArgs.fwd.camera.tanFovY, nullptr,
	    resource.geometryBuffer.data, resource.binningBuffer.data, resource.imageBuffer.data, roArgs.dL_dPixels,
	    dL_dmean2Ds, dL_dconics, rwArgs.dL_dSplats.opacities, dL_dcolors, rwArgs.dL_dSplats.means, dL_dcov3D,
	    rwArgs.dL_dSplats.shs, rwArgs.dL_dSplats.scales, rwArgs.dL_dSplats.rotates, false, perfQuery);

	cudaDeviceSynchronize();
}