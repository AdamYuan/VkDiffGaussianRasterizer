//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef VKGSRASTER_CAMERA_HPP
#define VKGSRASTER_CAMERA_HPP

#include <array>
#include <cinttypes>

namespace vkgsraster {

struct Camera {
	uint32_t width, height;
	float focalX, focalY;
	std::array<float, 3 * 3> viewMat;
	std::array<float, 3> pos;

	static float GetFocalFromTanFov(float tanFov, uint32_t dim) { return float(dim) * 0.5f / tanFov; }
	static float GetTanFovFromFocal(float focal, uint32_t dim) { return float(dim) * 0.5f / focal; }
	// actually, tanFov = tan(fov / 2)
};

} // namespace vkgsraster

#endif
