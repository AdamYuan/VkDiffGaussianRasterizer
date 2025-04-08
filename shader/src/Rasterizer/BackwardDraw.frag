#version 460
#extension GL_ARB_fragment_shader_interlock : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_quad : require
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

layout(rgba32f, binding = SIMG_IMAGE0_BINDING) coherent uniform image2D gMs_Rs;
layout(input_attachment_index = 0, binding = IATT_IMAGE0_BINDING) uniform subpassInput gDL_DPixels_Ts;

layout(pixel_interlock_ordered) in;

void main() {
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity);
	bool alphaDiscard = alpha < ALPHA_MIN;
	if (subgroupQuadAll(alphaDiscard))
		discard;

	// if (alphaDiscard)
	// 	alpha = 0;

	vec4 dL_dPixel_T = subpassLoad(gDL_DPixels_Ts);

	alpha = min(alpha, ALPHA_MAX);
	float oneMinusAlpha = 1.0 - alpha;
	vec3 alphaColor = alpha * gIn.color;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	beginInvocationInterlockARB();
	vec4 Mi_Ri = imageLoad(gMs_Rs, coord);  // M_i, R_i
	vec4 Mi_1_Ri_1 = Mi_Ri * oneMinusAlpha; // M_{i - 1}, R_{i - 1}
	Mi_1_Ri_1.xyz += alphaColor;
	imageStore(gMs_Rs, coord, Mi_1_Ri_1);
	endInvocationInterlockARB();

	vec3 dL_dPixel = dL_dPixel_T.xyz;
	float T = dL_dPixel_T.w;

	vec3 Mi = Mi_Ri.xyz;      // M_i
	float Ri_1 = Mi_1_Ri_1.w; // R_{i - 1}
	float Ti = T / Ri_1;      // T_i

	vec3 dL_dColor = dL_dPixel * (alpha * Ti);
	float dL_dAlpha = dot(dL_dPixel, (gIn.color - Mi) * Ti - gBgColor * T / oneMinusAlpha);

	SplatViewGeom splatViewGeom;
	splatViewGeom.conic = gIn.conic;
	splatViewGeom.mean2D = gIn.mean2D;
	splatViewGeom.opacity = gIn.opacity;

	Camera camera = loadCamera();

	SplatView dL_dSplatView;
	dL_dSplatView.color = dL_dColor;
	dL_dSplatView.geom = bwd_splatViewGeom2alpha(splatViewGeom, gl_FragCoord.xy, camera, dL_dAlpha);

	// if (alphaDiscard)
	// 	dL_dSplatView = zeroDL_DSplatView();

	bool callAtomicAdd;
	[[branch]]
	if (subgroupAllEqual(gIn.sortIdx)) {
		callAtomicAdd = subgroupElect();
		dL_dSplatView = subgroupReduceDL_DSplatView(dL_dSplatView);
	} else {
		callAtomicAdd = gl_SubgroupInvocationID == subgroupQuadBroadcast(gl_SubgroupInvocationID, 0);
		dL_dSplatView = quadReduceDL_DSplatView(dL_dSplatView);
	}

	if (callAtomicAdd)
		atomicAddDL_DSplatView(gIn.sortIdx, dL_dSplatView);
}
