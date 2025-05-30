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

layout(DL_DPIXEL_FORMAT_IDENTIFIER, binding = SIMG_DL_DPIXELS_BINDING) coherent uniform image2D gDL_DPixels;

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

	float dL_dPixel_pixel_i;
	vec3 dL_dPixel_Ti;

	bool TDiscard = false;

	beginInvocationInterlockARB();
	if (!pixelDiscard) {
		vec4 dL_dPixel_pixel_T = imageLoad(gDL_DPixels, coord);
		dL_dPixel_pixel_i = dL_dPixel_pixel_T.x;
		dL_dPixel_Ti = dL_dPixel_pixel_T.yzw;
		TDiscard = false;
		if (!TDiscard) {
			float dL_dPixel_pixel_i1 = dL_dPixel_pixel_i - dot(dL_dPixel_Ti, alphaColor);
			vec3 dL_dPixel_Ti1 = dL_dPixel_Ti * oneMinusAlpha;
			imageStore(gDL_DPixels, coord, vec4(dL_dPixel_pixel_i1, dL_dPixel_Ti1));
		}
	}
	endInvocationInterlockARB();

	pixelDiscard = pixelDiscard || TDiscard;
	if (subgroupQuadAll(pixelDiscard))
		return;

	SplatViewGeom splatViewGeom;
	splatViewGeom.conic = gIn.conic;
	splatViewGeom.mean2D = gIn.mean2D;
	splatViewGeom.opacity = gIn.opacity;

	Camera camera = loadCamera();

	SplatView dL_dSplatView;

	if (pixelDiscard)
		dL_dSplatView = zeroDL_DSplatView();
	else {
		vec3 dL_dColor = dL_dPixel_Ti * alpha;
		float dL_dAlpha = (dot(dL_dPixel_Ti, gIn.color) - dL_dPixel_pixel_i) / oneMinusAlpha;
		dL_dSplatView.color = dL_dColor;
		dL_dSplatView.geom = bwd_splatViewGeom2alpha(splatViewGeom, gl_FragCoord.xy, camera, G, dL_dAlpha);
	}

	// Should not perform atomicAdd on helper lanes
	uvec4 subgroupReduceMask = subgroupBallot(!pixelDiscard);

	[[branch]]
	if (subgroupAllEqual(gIn.sortIdx) && subgroupBallotBitCount(subgroupReduceMask) >= BALANCING_THRESHOLD) {
		dL_dSplatView = subgroupReduceDL_DSplatView(dL_dSplatView);
	} else {
		// subgroupReduceMask &= gl_SubgroupEqMask;
		subgroupReduceMask = mix(gl_SubgroupEqMask, uvec4(0), bvec4(pixelDiscard));
		subgroupReduceMask |= subgroupQuadSwapHorizontal(subgroupReduceMask);
		subgroupReduceMask |= subgroupQuadSwapVertical(subgroupReduceMask);
		dL_dSplatView = quadReduceDL_DSplatView(dL_dSplatView);
	}

	if (gl_SubgroupInvocationID == subgroupBallotFindLSB(subgroupReduceMask))
		atomicAddDL_DSplatView(gIn.sortIdx, dL_dSplatView);
}
