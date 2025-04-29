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
	struct Scene {
		std::string name;
		std::filesystem::path modelFilename;
		struct Entry {
			vkgsraster::Rasterizer::CameraArgs camera;
			std::string imageName;
		};
		std::vector<Entry> entries;
	};
	std::vector<Scene> scenes;

	static GSDataset Load(const std::filesystem::path &filename, uint32_t modelIteration = 7000);
	bool IsEmpty() const { return scenes.empty(); }

	void ResizeCamera(uint32_t width = 0, uint32_t height = 0);
};

#endif
