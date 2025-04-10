#version 460
#extension GL_ARB_fragment_shader_interlock : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_EXT_shader_quad_control : require

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

layout(rgba32f, binding = SIMG_IMAGE0_BINDING) coherent uniform image2D gColors_Ts;

layout(pixel_interlock_ordered, full_quads) in;

void main() {
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity);
#ifdef DEBUG_SUBGROUP
	if (subgroupAllEqual(gIn.sortIdx))
		alpha = 0;
#endif
	bool pixelDiscard = alpha < ALPHA_MIN;

	if (pixelDiscard)
		alpha = 0;

	alpha = min(alpha, ALPHA_MAX);
	float oneMinusAlpha = 1.0 - alpha;
	vec3 alphaColor = alpha * gIn.color;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	bool depthDiscard = pixelDiscard;

	beginInvocationInterlockARB();
	if (!pixelDiscard) {
		vec4 color_T = imageLoad(gColors_Ts, coord);
		depthDiscard = color_T.w >= T_MIN;
		if (depthDiscard) {
			color_T.xyz += alphaColor * color_T.w;
			color_T.w *= oneMinusAlpha;
			imageStore(gColors_Ts, coord, color_T);
		}
	}
	endInvocationInterlockARB();

	if (depthDiscard)
		discard; // Discard Depth Write
}
