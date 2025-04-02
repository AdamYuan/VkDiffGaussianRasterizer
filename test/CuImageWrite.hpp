//
// Created by adamyuan on 4/2/25.
//

#pragma once
#ifndef CUIMAGEWRITE_HPP
#define CUIMAGEWRITE_HPP

#include <filesystem>

struct CuImageWrite {
	static void Write(const std::filesystem::path &filename, const float *deviceColors, uint32_t width, uint32_t height);
};

#endif
