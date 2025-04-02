//
// Created by adamyuan on 4/1/25.
//

#include "CuTileRasterizer.hpp"

#include "GSModel.hpp"
#include "VkCuBuffer.hpp"
#include <rasterizer.h>

char *CuTileRasterizer::Resource::ResizeableBuffer::update(std::size_t updateSize) {
	if (updateSize > size) {
		if (data)
			cudaFree(data);
		cudaMalloc(&data, updateSize);
	}
	return data;
}

CuTileRasterizer::Camera CuTileRasterizer::Camera::Create(const VkGSRaster::Camera &vkCamera) {
	Camera camera{
	    .width = (int)vkCamera.width,
	    .height = (int)vkCamera.height,
	    .tanFovX = VkGSRaster::Camera::GetTanFovFromFocal(vkCamera.focalX, vkCamera.width),
	    .tanFovY = VkGSRaster::Camera::GetTanFovFromFocal(vkCamera.focalY, vkCamera.height),
	};

	using Mat4 = std::array<float, 16>;
	using Mat3 = std::array<float, 9>;
	const auto $4 = [](Mat4 &mat, uint32_t i, uint32_t j) -> float & { return mat[i + j * 4]; };
	const auto $3c = [](const Mat3 &mat, uint32_t i, uint32_t j) -> float { return mat[i + j * 3]; };

	Mat4 camProjMat{};
	{
		float zNear = 0.01f;
		float zFar = 100.0f;
		float top = camera.tanFovY * zNear;
		float bottom = -top;
		float right = camera.tanFovX * zNear;
		float left = -right;
		float zSign = 1.0f;
		$4(camProjMat, 0, 0) = 2.0f * zNear / (right - left);
		$4(camProjMat, 1, 1) = 2.0f * zNear / (top - bottom);
		$4(camProjMat, 0, 2) = (right + left) / (right - left);
		$4(camProjMat, 1, 2) = (top + bottom) / (top - bottom);
		$4(camProjMat, 3, 2) = zSign;
		$4(camProjMat, 2, 2) = zSign * zFar / (zFar - zNear);
		$4(camProjMat, 2, 3) = -(zFar * zNear) / (zFar - zNear);
	}

	Mat4 camViewMat{};
	{
		for (uint32_t i = 0; i < 3; ++i)
			for (uint32_t j = 0; j < 3; ++j)
				$4(camViewMat, i, j) = $3c(vkCamera.viewMat, i, j);
		for (uint32_t i = 0; i < 3; ++i)
			$4(camViewMat, i, 3) = -(vkCamera.pos[0] * $3c(vkCamera.viewMat, i, 0)   //
			                         + vkCamera.pos[1] * $3c(vkCamera.viewMat, i, 1) //
			                         + vkCamera.pos[2] * $3c(vkCamera.viewMat, i, 2));
		$4(camViewMat, 3, 3) = 1.0f;
	}

	float *deviceCamProjMat;
	cudaMalloc(&deviceCamProjMat, sizeof(Mat4));
	cudaMemcpy(deviceCamProjMat, camProjMat.data(), sizeof(Mat4), cudaMemcpyHostToDevice);
	camera.projMat = deviceCamProjMat;

	float *deviceCamViewMat;
	cudaMalloc(&deviceCamViewMat, sizeof(Mat4));
	cudaMemcpy(deviceCamViewMat, camViewMat.data(), sizeof(Mat4), cudaMemcpyHostToDevice);
	camera.viewMat = deviceCamViewMat;

	float *deviceCamPos;
	cudaMalloc(&deviceCamPos, sizeof(vkCamera.pos));
	cudaMemcpy(deviceCamPos, vkCamera.pos.data(), sizeof(vkCamera.pos), cudaMemcpyHostToDevice);
	camera.pos = deviceCamPos;

	return camera;
}

CuTileRasterizer::FwdArgs CuTileRasterizer::FwdArgs::Create(const VkGSRaster::Rasterizer::FwdROArgs &vkROArgs) {
	FwdArgs args = {
	    .camera = Camera::Create(vkROArgs.camera),
	    .splats =
	        {
	            .count = vkROArgs.splats.count,
	            .means = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pMeanBuffer)->GetCudaMappedPtr<float>(),
	            .scales = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pScaleBuffer)->GetCudaMappedPtr<float>(),
	            .rotates =
	                std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pRotateBuffer)->GetCudaMappedPtr<float>(),
	            .opacities =
	                std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pOpacityBuffer)->GetCudaMappedPtr<float>(),
	            .shs = std::static_pointer_cast<VkCuBuffer>(vkROArgs.splats.pSHBuffer)->GetCudaMappedPtr<float>(),
	        },
	};

	float *deviceBgColor;
	cudaMalloc(&deviceBgColor, sizeof(vkROArgs.bgColor));
	cudaMemcpy(deviceBgColor, vkROArgs.bgColor.data(), sizeof(vkROArgs.bgColor), cudaMemcpyHostToDevice);

	args.bgColor = deviceBgColor;

	cudaMalloc(&args.outColor, sizeof(float) * 3 * vkROArgs.camera.width * vkROArgs.camera.height);

	return args;
}

void CuTileRasterizer::Forward(const FwdArgs &args, Resource &resource) {
	CudaRasterizer::Rasterizer::forward(
	    [&](std::size_t size) { return resource.geometryBuffer.update(size); },
	    [&](std::size_t size) { return resource.binningBuffer.update(size); },
	    [&](std::size_t size) { return resource.imageBuffer.update(size); }, (int)args.splats.count, GSModel::kSHDegree,
	    GSModel::kSHSize, args.bgColor, args.camera.width, args.camera.height, args.splats.means, args.splats.shs,
	    nullptr, args.splats.opacities, args.splats.scales, 1.0f, args.splats.rotates, nullptr, args.camera.viewMat,
	    args.camera.projMat, args.camera.pos, args.camera.tanFovX, args.camera.tanFovY, false, args.outColor);
}