//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef GSDATASET_HPP
#define GSDATASET_HPP

#include "../src/Rasterizer.hpp"

#include <filesystem>
#include <string>
#include <vector>

struct GSDataset {
	struct Entry {
		vkgsraster::Rasterizer::CameraArgs camera;
		std::string imageName;
	};
	std::vector<Entry> entries;

	static GSDataset Load(const std::filesystem::path &filename);
	bool IsEmpty() const { return entries.empty(); }
};

#endif
