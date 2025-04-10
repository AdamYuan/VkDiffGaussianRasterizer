#include <array>
#include <cinttypes>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
	--argc, ++argv;
	if (argc != 1) {
		printf("./tri_strip [vertNum]\n");
		return -1;
	}
	uint32_t vertNum = std::stoi(argv[0]);
	double halfAngle = M_PI / double(vertNum), angle = 2.0 * halfAngle;
	double vertDist = 1.0 / std::cos(halfAngle);

	std::vector<std::array<float, 2>> verts(vertNum);

	for (double theta = halfAngle; std::array<float, 2> & vert : verts) {
		vert[0] = std::cos(theta) * vertDist;
		vert[1] = std::sin(theta) * vertDist;
		theta += angle;
	}

	printf("#define VERT_NUM %d\n", vertNum);
	printf("const vec2[VERT_NUM] kVerts = {\n");
	printf("\tvec2(%lf, %lf),\n", verts[0][0], verts[0][1]);

	for (uint32_t vertOfst = 1; vertOfst < (vertNum + 1) / 2; ++vertOfst) {
		printf("\tvec2(%lf, %lf),\n", verts[vertOfst][0], verts[vertOfst][1]);
		printf("\tvec2(%lf, %lf),\n", verts[vertNum - vertOfst][0], verts[vertNum - vertOfst][1]);
	}

	if (vertNum % 2 == 0)
		printf("\tvec2(%lf, %lf),\n", verts[vertNum / 2][0], verts[vertNum / 2][1]);
	printf("};\n");

	return 0;
}
