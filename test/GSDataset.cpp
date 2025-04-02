//
// Created by adamyuan on 3/27/25.
//

#include "GSDataset.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

GSDataset GSDataset::Load(const std::filesystem::path &filename) {
	std::ifstream fin{filename};
	auto json = nlohmann::json::parse(fin);

	GSDataset dataset;
	dataset.entries.reserve(json.size());

	for (const auto &element : json) {
		Entry entry;
		entry.imageName = element["img_name"].get<std::string>();
		entry.camera.width = element["width"].get<uint32_t>();
		entry.camera.height = element["height"].get<uint32_t>();
		entry.camera.focalX = element["fx"].get<float>();
		entry.camera.focalY = element["fy"].get<float>();
		entry.camera.pos = element["position"].get<std::array<float, 3>>();
		std::ranges::copy(element["rotation"][0].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 0);
		std::ranges::copy(element["rotation"][1].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 3);
		std::ranges::copy(element["rotation"][2].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 6);
		dataset.entries.push_back(entry);
	}
	return dataset;
}