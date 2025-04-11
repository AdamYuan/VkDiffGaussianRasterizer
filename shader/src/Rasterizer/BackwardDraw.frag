#version 460
#extension GL_ARB_fragment_shader_interlock : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_EXT_maximal_reconvergence : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_quad_control : require

#define RASTERIZER_ATOMICADD_DL_DSPLAT_VIEW
#define RASTERIZER_REDUCE_DL_DSPLAT_VIEW
#include "Common.glsl"

in bIn {
	layout(location = 0) flat vec3 color;
	layout(location = 1) flat vec3 conic;
	layout(location = 2) flat vec2 mean2D;
	layout(location = 3) flat float opacity;
	layout(location = 4) flat uint sortIdx;
	layout(location = 5) noperspective vec2 quadPos;
}
gIn;

layout(rgba32f, binding = SIMG_IMAGE0_BINDING) coherent uniform image2D gPixels_Ts;
layout(input_attachment_index = 0, binding = IATT_IMAGE0_BINDING) uniform subpassInput gDL_DPixels_Ts;

layout(pixel_interlock_ordered, full_quads, early_fragment_tests) in;

#define BALANCING_THRESHOLD 16

void main() {
	float G;
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity, G);
	bool pixelDiscard = alpha < ALPHA_MIN || gl_HelperInvocation;
	bool quadDiscard = subgroupQuadAll(pixelDiscard);
	if (quadDiscard)
		discard;

	if (pixelDiscard)
		alpha = 0;

	vec4 dL_dPixel_T = subpassLoad(gDL_DPixels_Ts);

	alpha = min(alpha, ALPHA_MAX);
	float oneMinusAlpha = 1.0 - alpha;
	vec3 alphaColor = alpha * gIn.color;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	float T_i, T_i1; // T_i, T_{i+1}
	vec3 pixel_i, pixel_i1;

	beginInvocationInterlockARB();
	if (!pixelDiscard) {
		vec4 pixel_T = imageLoad(gPixels_Ts, coord);
		pixel_i = pixel_T.xyz;
		T_i = pixel_T.w;
		pixel_i1 = pixel_i - T_i * alphaColor;
		T_i1 = T_i * oneMinusAlpha;
		imageStore(gPixels_Ts, coord, vec4(pixel_i1, T_i1));
	}
	endInvocationInterlockARB();

	if (quadDiscard)
		return;

	vec3 dL_dPixel = dL_dPixel_T.xyz;
	float T = dL_dPixel_T.w;

	vec3 dL_dColor = dL_dPixel * (alpha * T_i);
	float dL_dAlpha = dot(dL_dPixel, (gIn.color - (pixel_i1 / T_i1)) * T_i - gBgColor * T / oneMinusAlpha);

	SplatViewGeom splatViewGeom;
	splatViewGeom.conic = gIn.conic;
	splatViewGeom.mean2D = gIn.mean2D;
	splatViewGeom.opacity = gIn.opacity;

	Camera camera = loadCamera();

	SplatView dL_dSplatView;

	if (pixelDiscard)
		dL_dSplatView = zeroDL_DSplatView();
	else {
		dL_dSplatView.color = dL_dColor;
		dL_dSplatView.geom = bwd_splatViewGeom2alpha(splatViewGeom, gl_FragCoord.xy, camera, G, dL_dAlpha);
	}

	// Should not perform atomicAdd on helper lanes
	uvec4 subgroupReduceMask = subgroupBallot(!pixelDiscard);

	bool callAtomicAdd;
	[[branch]]
	if (subgroupAllEqual(gIn.sortIdx) && subgroupBallotBitCount(subgroupReduceMask) >= BALANCING_THRESHOLD) {
		callAtomicAdd = gl_SubgroupInvocationID == subgroupBallotFindLSB(subgroupReduceMask);
		dL_dSplatView = subgroupReduceDL_DSplatView(dL_dSplatView);
	} else {
		uvec4 quadAtomicAddMask = subgroupReduceMask & gl_SubgroupEqMask;
		quadAtomicAddMask |= subgroupQuadSwapHorizontal(quadAtomicAddMask);
		quadAtomicAddMask |= subgroupQuadSwapVertical(quadAtomicAddMask);

		callAtomicAdd = gl_SubgroupInvocationID == subgroupBallotFindLSB(quadAtomicAddMask);
		dL_dSplatView = quadReduceDL_DSplatView(dL_dSplatView);
	}

	if (callAtomicAdd)
		atomicAddDL_DSplatView(gIn.sortIdx, dL_dSplatView);
}
