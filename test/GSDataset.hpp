//
// Created by adamyuan on 3/27/25.
//

#pragma once
#ifndef GSDATASET_HPP
#define GSDATASET_HPP

#include "../src/Rasterizer.hpp"

#include <algorithm>
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

	uint32_t GetMaxSplatCount() const;

	void ResizeCamera(uint32_t width = 0, uint32_t height = 0);
	void RandomCrop(auto &&randGen, uint32_t entriesPerScene = 0) {
		for (auto &scene : scenes) {
			if (scene.entries.size() <= entriesPerScene)
				continue;
			std::shuffle(scene.entries.begin(), scene.entries.end(), randGen);
			scene.entries.erase(scene.entries.begin() + entriesPerScene, scene.entries.end());
		}
	}
};

#endif
