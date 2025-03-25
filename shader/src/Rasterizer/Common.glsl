#ifndef RASTERIZER_COMMON_GLSL
#define RASTERIZER_COMMON_GLSL

#extension GL_EXT_control_flow_attributes : require

#include "Math.glsl"
#include "Size.glsl"

layout(std430, push_constant) uniform bPushConstant {
	vec3 gBgColor;
	uint gSplatCount;
	vec2 gCamFocal;
	uvec2 gCamResolution;
	vec3 gCamPos;
	float gCamViewMat[9];
};

Camera loadCamera() {
	Camera cam;
	cam.focal = gCamFocal;
	cam.resolution = gCamResolution;
	cam.pos = gCamPos;
	cam.viewMat = mat3(gCamViewMat[0], gCamViewMat[1], gCamViewMat[2], //
	                   gCamViewMat[3], gCamViewMat[4], gCamViewMat[5], //
	                   gCamViewMat[6], gCamViewMat[7], gCamViewMat[8]);
	return cam;
}

layout(binding = B_SORT_COUNT_BINDING) uniform bSortCount {
	layout(offset = SORT_COUNT_BUFFER_OFFSET) uint gSplatSortCount;
};

struct Vec3Std430 {
	float vec[3];
};
struct SHStd430 {
	float vec[3 * SH_SIZE];
};

// Splat Load
#ifdef RASTERIZER_LOAD_SPLAT
layout(std430, binding = B_MEANS_BINDING) readonly buffer bMeans { Vec3Std430 gMeans[]; };
layout(std430, binding = B_SCALES_BINDING) readonly buffer bScales { Vec3Std430 gScales[]; };
layout(std430, binding = B_ROTATES_BINDING) readonly buffer bRotates { vec4 gRotates[]; };
layout(std430, binding = B_OPACITIES_BINDING) readonly buffer bOpacities { float gOpacities[]; };
layout(std430, binding = B_SHS_BINDING) readonly buffer bSHs { SHStd430 gSHs[]; };
Splat loadSplat(uint splatIdx) {
	Vec3Std430 mean = gMeans[splatIdx];
	Vec3Std430 scale = gScales[splatIdx];
	vec4 quat = gRotates[splatIdx];
	float opacity = gOpacities[splatIdx];
	SHStd430 sh = gSHs[splatIdx];

	Splat splat;
	splat.geom.quat = quat;
	splat.geom.scale = vec3(scale.vec[0], scale.vec[1], scale.vec[2]);
	splat.geom.mean = vec3(mean.vec[0], mean.vec[1], mean.vec[2]);
	splat.geom.opacity = opacity;

	[[unroll]]
	for (uint i = 0; i < SH_SIZE; ++i)
		splat.sh.data[i] = vec3(sh.vec[i * 3 + 0], sh.vec[i * 3 + 1], sh.vec[i * 3 + 2]);
	return splat;
}
#endif

// SplatView Load
#ifdef RASTERIZER_LOAD_SPLAT_VIEW
#ifndef RASTERIZER_LOAD_SPLAT
layout(std430, binding = B_OPACITIES_BINDING) readonly buffer bOpacities { float gOpacities[]; };
#endif
layout(std430, binding = B_COLORS_MEAN2DXS_BINDING) readonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = B_CONICS_MEAN2DYS_BINDING) readonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
SplatView loadSplatView(uint splatIdx) {
	float opacity = gOpacities[splatIdx];
	vec4 colors_mean2Dxs = gColors_Mean2DXs[splatIdx];
	vec4 conics_mean2Dys = gConics_Mean2DYs[splatIdx];

	SplatView splatView;
	splatView.geom.conic = conics_mean2Dys.xyz;
	splatView.geom.opacity = opacity;
	splatView.geom.mean2D = vec2(colors_mean2Dxs.w, conics_mean2Dys.w);
	splatView.color = colors_mean2Dxs.xyz;
	return splatView;
}
#endif

// SplatQuad Load
#ifdef RASTERIZER_LOAD_SPLAT_QUAD
layout(std430, binding = B_QUADS_BINDING) readonly buffer bQuads { vec4 gQuads[]; };
SplatQuad loadSplatQuad(uint splatIdx) {
	vec4 data = gQuads[splatIdx];
	SplatQuad quad;
	quad.axis1 = data.xy;
	quad.axis2 = data.zw;
	return quad;
}
#endif

// SplatView Store
#ifdef RASTERIZER_STORE_SPLAT_VIEW
layout(std430, binding = B_COLORS_MEAN2DXS_BINDING) writeonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = B_CONICS_MEAN2DYS_BINDING) writeonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
void storeSplatView(uint splatIdx, SplatView splatView) {
	gColors_Mean2DXs[splatIdx] = vec4(splatView.color, splatView.geom.mean2D.x);
	gConics_Mean2DYs[splatIdx] = vec4(splatView.geom.conic, splatView.geom.mean2D.y);
}
#endif

// SplatQuad Store
#ifdef RASTERIZER_STORE_SPLAT_QUAD
layout(std430, binding = B_QUADS_BINDING) writeonly buffer bQuads { vec4 gQuads[]; };
void storeSplatQuad(uint splatIdx, SplatQuad quad) { gQuads[splatIdx] = vec4(quad.axis1, quad.axis2); }
#endif

#endif
