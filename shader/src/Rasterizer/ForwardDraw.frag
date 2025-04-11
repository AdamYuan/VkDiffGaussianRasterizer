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

layout(PIXEL_T_FORMAT_IDENTIFIER, binding = SIMG_PIXELS_TS_BINDING) coherent uniform image2D gPixels_Ts;

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
		vec4 pixel_T = imageLoad(gPixels_Ts, coord);
		depthDiscard = pixel_T.w >= T_MIN;
		if (depthDiscard) {
			pixel_T.xyz += alphaColor * pixel_T.w;
			pixel_T.w *= oneMinusAlpha;
			imageStore(gPixels_Ts, coord, pixel_T);
		}
	}
	endInvocationInterlockARB();

	if (depthDiscard)
		discard; // Discard Depth Write
}
