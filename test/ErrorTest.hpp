//
// Created by adamyuan on 4/30/25.
//

#pragma once
#ifndef TEST_ERRORTEST_HPP
#define TEST_ERRORTEST_HPP

#include "CuTileRasterizer.hpp"
#include "GSModel.hpp"
#include <span>

struct GSGradient {
	struct Error {
		double mean, scale, rotate, opacity, sh0;

		Error &operator+=(const Error &r) {
			mean += r.mean;
			scale += r.scale;
			rotate += r.rotate;
			opacity += r.opacity;
			sh0 += r.sh0;
			return *this;
		}
		Error &operator/=(double r) {
			mean /= r;
			scale /= r;
			rotate /= r;
			opacity /= r;
			sh0 /= r;
			return *this;
		}
	};

	using Mean = GSModel::Mean;
	using Scale = GSModel::Scale;
	using Rotate = GSModel::Rotate;
	using Opacity = GSModel::Opacity;
	using SH0 = std::array<float, 3>;

	uint32_t splatCount{};
	std::vector<Mean> means;
	std::vector<Scale> scales;
	std::vector<Rotate> rotates;
	std::vector<Opacity> opacities;
	std::vector<SH0> sh0s;

	void Update(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount);
	Error GetError(const GSGradient &r, auto &&errorFunc) const {
		const auto getError = [&errorFunc]<typename T>(const std::vector<T> &yHat, const std::vector<T> &y) {
			static_assert(sizeof(T) % sizeof(float) == 0);
			std::span yHatFlt{reinterpret_cast<const float *>(yHat.data()),
			                  reinterpret_cast<const float *>(yHat.data() + yHat.size())};
			std::span yFlt{reinterpret_cast<const float *>(y.data()), //
			               reinterpret_cast<const float *>(y.data() + y.size())};
			return errorFunc(yHatFlt, yFlt);
		};

		Error rrmse{};
		rrmse.mean = getError(means, r.means);
		rrmse.scale = getError(scales, r.scales);
		rrmse.rotate = getError(rotates, r.rotates);
		rrmse.opacity = getError(opacities, r.opacities);
		rrmse.sh0 = getError(sh0s, r.sh0s);
		return rrmse;
	}
};

#endif
