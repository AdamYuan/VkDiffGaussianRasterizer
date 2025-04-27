#version 460

#define DEBUG_SUBGROUP

#ifdef DEBUG_SUBGROUP
#extension GL_KHR_shader_subgroup_vote : require
#endif

#define RASTERIZER_VERBOSE
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
	vec3 color = gIn.color;
	bool cohesion = subgroupAllEqual(gIn.sortIdx);
	bool pixelDiscard = alpha < ALPHA_MIN;
	if (pixelDiscard)
		discard;

#ifdef DEBUG_SUBGROUP
	if (!cohesion) {
		color = vec3(1, 0, 0);
		// alpha = 1.0;
	}
#endif

	VERBOSE_ADD(FragmentCount);
	if (cohesion)
		VERBOSE_ADD(CoherentFragmentCount);

	alpha = min(alpha, ALPHA_MAX);
	gOutFragColor = vec4(color, alpha);
}
