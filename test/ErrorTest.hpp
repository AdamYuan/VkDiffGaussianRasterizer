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
	using Mean = GSModel::Mean;
	using Scale = GSModel::Scale;
	using Rotate = GSModel::Rotate;
	using Opacity = GSModel::Opacity;
	using SH = GSModel::SH;

	uint32_t splatCount{};
	std::vector<float> values;

	std::size_t GetValueCount() const {
		return splatCount *
		       ((sizeof(Mean) + sizeof(Scale) + sizeof(Rotate) + sizeof(Opacity) + sizeof(SH)) / sizeof(float));
	}
	void Update(const CuTileRasterizer::SplatArgs &splats, uint32_t splatCount);
	auto GetError(const GSGradient &r, auto &&errorFunc) const {
		return errorFunc({values.data(), GetValueCount()}, {r.values.data(), GetValueCount()});
	}
};

#endif
