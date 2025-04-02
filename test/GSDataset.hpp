//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef GSDATASET_HPP
#define GSDATASET_HPP

#include "../src/Camera.hpp"
#include <filesystem>
#include <string>
#include <vector>

struct GSDataset {
	struct Entry {
		VkGSRaster::Camera camera;
		std::string imageName;
	};
	std::vector<Entry> entries;

	static GSDataset Load(const std::filesystem::path &filename);
};

#endif
