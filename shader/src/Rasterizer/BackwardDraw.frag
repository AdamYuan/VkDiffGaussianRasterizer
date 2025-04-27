#version 460
#extension GL_ARB_fragment_shader_interlock : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_EXT_maximal_reconvergence : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_quad_control : require

#define RASTERIZER_ATOMICADD_DL_DSPLAT_VIEW
#define RASTERIZER_REDUCE_DL_DSPLAT_VIEW
#define RASTERIZER_VERBOSE
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

layout(PIXEL_T_FORMAT_IDENTIFIER, binding = SIMG_PIXELS_TS_BINDING) coherent uniform image2D gPixels_Ts;
layout(input_attachment_index = 0, binding = IATT_DL_DPIXELS_BINDING) uniform subpassInput gDL_DPixels;

layout(pixel_interlock_ordered, full_quads) in;

#define BALANCING_THRESHOLD 16

void main() {
	float G;
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity, G);
	bool pixelDiscard = alpha < ALPHA_MIN || gl_HelperInvocation;

	if (pixelDiscard)
		alpha = 0;

	alpha = min(alpha, ALPHA_MAX);
	float oneMinusAlpha = 1.0 - alpha;
	vec3 alphaColor = alpha * gIn.color;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	float T_i;
	vec3 pixel_i;

	bool TDiscard = false;

	beginInvocationInterlockARB();
	if (!pixelDiscard) {
		vec4 pixel_T = imageLoad(gPixels_Ts, coord);
		pixel_i = pixel_T.xyz;
		T_i = pixel_T.w;
		TDiscard = T_i < T_MIN;
		if (!TDiscard) {
			float T_i1 = T_i * oneMinusAlpha;
			vec3 pixel_i1 = pixel_i - T_i * alphaColor;
			imageStore(gPixels_Ts, coord, vec4(pixel_i1, T_i1));
		}
	}
	endInvocationInterlockARB();

	pixelDiscard = pixelDiscard || TDiscard;
	if (subgroupQuadAll(pixelDiscard))
		return;

	vec3 dL_dPixel = subpassLoad(gDL_DPixels).xyz;

	vec3 dL_dColor = dL_dPixel * (alpha * T_i);
	float dL_dAlpha = dot(dL_dPixel, gIn.color * T_i - pixel_i) / oneMinusAlpha;

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
	bool subgroupCoherent = subgroupAllEqual(gIn.sortIdx);

	[[branch]]
	if (subgroupCoherent && subgroupBallotBitCount(subgroupReduceMask) >= BALANCING_THRESHOLD) {
		dL_dSplatView = subgroupReduceDL_DSplatView(dL_dSplatView);
	} else {
		subgroupReduceMask &= gl_SubgroupEqMask;
		subgroupReduceMask |= subgroupQuadSwapHorizontal(subgroupReduceMask);
		subgroupReduceMask |= subgroupQuadSwapVertical(subgroupReduceMask);
		dL_dSplatView = quadReduceDL_DSplatView(dL_dSplatView);
	}

	if (gl_SubgroupInvocationID == subgroupBallotFindLSB(subgroupReduceMask)) {
		atomicAddDL_DSplatView(gIn.sortIdx, dL_dSplatView);
		VERBOSE_ADD(AtomicAddCount, 1u);
		uint count = subgroupBallotBitCount(subgroupReduceMask);
		VERBOSE_ADD(FragmentCount, count);
		if (subgroupCoherent)
			VERBOSE_ADD(CoherentFragmentCount, count);
	}
}
