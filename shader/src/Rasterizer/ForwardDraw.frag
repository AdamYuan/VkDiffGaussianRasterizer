#version 460

// #define DEBUG_SUBGROUP

#ifdef DEBUG_SUBGROUP
#extension GL_KHR_shader_subgroup_vote : require
#endif

#include "Common.glsl"

in bIn {
	layout(location = 0) flat float opacity;
	layout(location = 1) flat vec3 color;
	layout(location = 2) noperspective vec2 quadPos;
#ifdef DEBUG_SUBGROUP
	layout(location = 3) flat uint sortIdx;
#endif
}
gIn;

layout(location = 0) out vec4 gOutFragColor;

void main() {
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity);
#ifdef DEBUG_SUBGROUP
	if (subgroupAllEqual(gIn.sortIdx))
		alpha = 0;
#endif
	bool pixelDiscard = alpha < ALPHA_MIN;
	if (pixelDiscard)
		discard;

	alpha = min(alpha, ALPHA_MAX);
	gOutFragColor = vec4(gIn.color, alpha);
}
