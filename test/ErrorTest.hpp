//
// Created by adamyuan on 4/30/25.
//

#pragma once
#ifndef TEST_ERRORTEST_HPP
#define TEST_ERRORTEST_HPP

#include "CuTileRasterizer.hpp"
#include "GSModel.hpp"

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
	// Mean Relative Error
	Error GetMRE(const GSGradient &r) const;
};

#endif
