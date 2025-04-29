//
// Created by adamyuan on 3/27/25.
//

#include "GSDataset.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

GSDataset GSDataset::Load(const std::filesystem::path &filename, uint32_t modelIteration) {
	const auto loadScene = [](const std::filesystem::path &sceneFilename, uint32_t modelIteration) -> Scene {
		auto jsonFilename = sceneFilename / "cameras.json";
		auto modelFilename =
		    sceneFilename / "point_cloud" / ("iteration_" + std::to_string(modelIteration)) / "point_cloud.ply";

		Scene scene{
		    .name = sceneFilename.filename().string(),
		    .modelFilename = modelFilename,
		};

		std::ifstream jsonFin{jsonFilename};
		auto json = nlohmann::json::parse(jsonFin);

		scene.entries.reserve(json.size());

		for (const auto &element : json) {
			Scene::Entry entry;
			entry.imageName = element["img_name"].get<std::string>();
			entry.camera.width = element["width"].get<uint32_t>();
			entry.camera.height = element["height"].get<uint32_t>();
			entry.camera.focalX = element["fx"].get<float>();
			entry.camera.focalY = element["fy"].get<float>();
			entry.camera.pos = element["position"].get<std::array<float, 3>>();
			std::ranges::copy(element["rotation"][0].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 0);
			std::ranges::copy(element["rotation"][1].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 3);
			std::ranges::copy(element["rotation"][2].get<std::array<float, 3>>(), entry.camera.viewMat.data() + 6);
			scene.entries.push_back(entry);
		}

		return scene;
	};

	GSDataset dataset;

	if (std::filesystem::is_directory(filename))
		dataset.scenes.push_back(loadScene(filename, modelIteration));
	else {
		std::ifstream fin{filename};
		std::string sceneName;
		while (std::getline(fin, sceneName)) {
			auto sceneFilename = std::filesystem::path{filename}.replace_filename(sceneName);
			dataset.scenes.push_back(loadScene(sceneFilename, modelIteration));
		}
	}

	return dataset;
}

void GSDataset::ResizeCamera(uint32_t width, uint32_t height) {
	if (width == 0 && height == 0)
		return;
	for (auto &scene : scenes) {
		for (auto &entry : scene.entries) {
			float focalRatio;
			if (width && height) {
				entry.camera.width = width;
				entry.camera.height = height;
				focalRatio = float(width) / float(entry.camera.width);
			} else if (width) {
				entry.camera.height = entry.camera.height * width / entry.camera.width;
				entry.camera.width = width;
				focalRatio = float(width) / float(entry.camera.width);
			} else {
				entry.camera.width = entry.camera.width * height / entry.camera.height;
				entry.camera.height = height;
				focalRatio = float(height) / float(entry.camera.height);
			}
			entry.camera.focalX *= focalRatio;
			entry.camera.focalY *= focalRatio;
		}
	}
}