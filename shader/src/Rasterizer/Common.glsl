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
	cam.viewMat = mat3(gCamViewMat[0], gCamViewMat[3], gCamViewMat[6], //
	                   gCamViewMat[1], gCamViewMat[4], gCamViewMat[7], //
	                   gCamViewMat[2], gCamViewMat[5], gCamViewMat[8]);
	return cam;
}

layout(binding = UBUF_SORT_COUNT_BINDING) uniform bSortCount {
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
layout(std430, binding = SBUF_MEANS_BINDING) readonly buffer bMeans { Vec3Std430 gMeans[]; };
layout(std430, binding = SBUF_SCALES_BINDING) readonly buffer bScales { Vec3Std430 gScales[]; };
layout(std430, binding = SBUF_ROTATES_BINDING) readonly buffer bRotates { vec4 gRotates[]; };
layout(std430, binding = SBUF_OPACITIES_BINDING) readonly buffer bOpacities { float gOpacities[]; };
layout(std430, binding = SBUF_SHS_BINDING) readonly buffer bSHs { SHStd430 gSHs[]; };
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
layout(std430, binding = SBUF_COLORS_MEAN2DXS_BINDING) readonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = SBUF_CONICS_MEAN2DYS_BINDING) readonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
layout(std430, binding = SBUF_VIEW_OPACITIES_BINDING) readonly buffer bViewOpacities { float gViewOpacities[]; };
SplatView loadSplatView(uint sortIdx) {
	float opacity = gViewOpacities[sortIdx];
	vec4 colors_mean2Dxs = gColors_Mean2DXs[sortIdx];
	vec4 conics_mean2Dys = gConics_Mean2DYs[sortIdx];

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
layout(std430, binding = SBUF_QUADS_BINDING) readonly buffer bQuads { vec4 gQuads[]; };
SplatQuad loadSplatQuad(uint sortIdx) {
	vec4 data = gQuads[sortIdx];
	SplatQuad quad;
	quad.axis1 = data.xy;
	quad.axis2 = data.zw;
	return quad;
}
#endif

// SplatView Store
#ifdef RASTERIZER_STORE_SPLAT_VIEW
layout(std430, binding = SBUF_COLORS_MEAN2DXS_BINDING) writeonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = SBUF_CONICS_MEAN2DYS_BINDING) writeonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
layout(std430, binding = SBUF_VIEW_OPACITIES_BINDING) writeonly buffer bViewOpacities { float gViewOpacities[]; };
void storeSplatView(uint sortIdx, SplatView splatView) {
	gColors_Mean2DXs[sortIdx] = vec4(splatView.color, splatView.geom.mean2D.x);
	gConics_Mean2DYs[sortIdx] = vec4(splatView.geom.conic, splatView.geom.mean2D.y);
	gViewOpacities[sortIdx] = splatView.geom.opacity;
}
#endif

// SplatQuad Store
#ifdef RASTERIZER_STORE_SPLAT_QUAD
layout(std430, binding = SBUF_QUADS_BINDING) writeonly buffer bQuads { vec4 gQuads[]; };
void storeSplatQuad(uint sortIdx, SplatQuad quad) { gQuads[sortIdx] = vec4(quad.axis1, quad.axis2); }
#endif

#if defined(RASTERIZER_LOAD_PIXEL) || defined(RASTERIZER_STORE_PIXEL)
uint pixelCoord2Idx(uvec2 pixelCoord) { return pixelCoord.x + gCamResolution.x * pixelCoord.y; }
uvec2 pixelIdx2Coord(uint pixelIdx) { return uvec2(pixelIdx % gCamResolution.x, pixelIdx / gCamResolution.x); }
bool validPixelCoord(uvec2 pixelCoord) { return all(lessThan(pixelCoord, gCamResolution)); }
#endif

// Pixel Load
#ifdef RASTERIZER_LOAD_PIXEL
layout(std430, binding = SBUF_PIXELS_BINDING) readonly buffer bPixels { float gPixels[]; };
vec3 loadPixel(uint pixelIdx) {
	uint pixelCount = gCamResolution.x * gCamResolution.y;
	return vec3(gPixels[pixelIdx], gPixels[pixelCount + pixelIdx], gPixels[pixelCount * 2 + pixelIdx]);
}
vec3 loadPixel(uvec2 pixelCoord) { return loadPixel(pixelCoord2Idx(pixelCoord)); }
#endif

// Pixel Store
#ifdef RASTERIZER_STORE_PIXEL
layout(std430, binding = SBUF_PIXELS_BINDING) writeonly buffer bPixels { float gPixels[]; };
void storePixel(uint pixelIdx, vec3 color) {
	uint pixelCount = gCamResolution.x * gCamResolution.y;
	gPixels[pixelIdx] = color.x;
	gPixels[pixelCount + pixelIdx] = color.y;
	gPixels[pixelCount * 2 + pixelIdx] = color.z;
}
void storePixel(uvec2 pixelCoord, vec3 color) { storePixel(pixelCoord2Idx(pixelCoord), color); }
#endif

// DL_DSplat Store
#ifdef RASTERIZER_STORE_DL_DSPLAT
layout(std430, binding = SBUF_DL_DMEANS_BINDING) writeonly buffer bDL_DMeans { Vec3Std430 gDL_DMeans[]; };
layout(std430, binding = SBUF_DL_DSCALES_BINDING) writeonly buffer bDL_DScales { Vec3Std430 gDL_DScales[]; };
layout(std430, binding = SBUF_DL_DROTATES_BINDING) writeonly buffer bDL_DRotates { vec4 gDL_DRotates[]; };
layout(std430, binding = SBUF_DL_DOPACITIES_BINDING) writeonly buffer bDL_DOpacities { float gDL_DOpacities[]; };
layout(std430, binding = SBUF_DL_DSHS_BINDING) writeonly buffer bDL_DSHs { SHStd430 gDL_DSHs[]; };
void storeDL_DSplat(uint splatIdx, Splat dL_dSplat) {
	gDL_DRotates[splatIdx] = dL_dSplat.geom.quat;
	gDL_DMeans[splatIdx] = Vec3Std430(float[3](dL_dSplat.geom.mean.x, dL_dSplat.geom.mean.y, dL_dSplat.geom.mean.z));
	gDL_DScales[splatIdx] =
	    Vec3Std430(float[3](dL_dSplat.geom.scale.x, dL_dSplat.geom.scale.y, dL_dSplat.geom.scale.z));
	gDL_DOpacities[splatIdx] = dL_dSplat.geom.opacity;

	SHStd430 sh;
	[[unroll]]
	for (uint i = 0; i < SH_SIZE; ++i) {
		sh.vec[i * 3 + 0] = dL_dSplat.sh.data[i].x;
		sh.vec[i * 3 + 1] = dL_dSplat.sh.data[i].y;
		sh.vec[i * 3 + 2] = dL_dSplat.sh.data[i].z;
	}
	gDL_DSHs[splatIdx] = sh;
}
#endif

// DL_DSplatView Load
#ifdef RASTERIZER_LOAD_DL_DSPLAT_VIEW
layout(std430, binding = SBUF_DL_DCOLORS_MEAN2DXS_BINDING) readonly buffer bDL_DColors_Mean2DXs {
	vec4 gDL_DColors_Mean2DXs[];
};
layout(std430, binding = SBUF_DL_DCONICS_MEAN2DYS_BINDING) readonly buffer bDL_DConics_Mean2DYs {
	vec4 gDL_DConics_Mean2DYs[];
};
layout(std430, binding = SBUF_DL_DVIEW_OPACITIES_BINDING) readonly buffer bDL_DViewOpacities {
	float gDL_DViewOpacities[];
};
SplatView loadDL_DSplatView(uint sortIdx) {
	vec4 dL_dColors_mean2Dxs = gDL_DColors_Mean2DXs[sortIdx];
	vec4 dL_dConics_mean2Dys = gDL_DConics_Mean2DYs[sortIdx];
	float dL_dViewOpacity = gDL_DViewOpacities[sortIdx];

	SplatView dL_dSplatView;
	dL_dSplatView.geom.conic = dL_dConics_mean2Dys.xyz;
	dL_dSplatView.geom.opacity = dL_dViewOpacity;
	dL_dSplatView.geom.mean2D = vec2(dL_dColors_mean2Dxs.w, dL_dConics_mean2Dys.w);
	dL_dSplatView.color = dL_dColors_mean2Dxs.xyz;
	return dL_dSplatView;
}
#endif

// DL_DSplatView Clear
#ifdef RASTERIZER_CLEAR_DL_DSPLAT_VIEW
layout(std430, binding = SBUF_DL_DCOLORS_MEAN2DXS_BINDING) writeonly buffer bDL_DColors_Mean2DXs {
	vec4 gDL_DColors_Mean2DXs[];
};
layout(std430, binding = SBUF_DL_DCONICS_MEAN2DYS_BINDING) writeonly buffer bDL_DConics_Mean2DYs {
	vec4 gDL_DConics_Mean2DYs[];
};
layout(std430, binding = SBUF_DL_DVIEW_OPACITIES_BINDING) writeonly buffer bDL_DViewOpacities {
	float gDL_DViewOpacities[];
};
void clearDL_DSplatView(uint sortIdx) {
	gDL_DColors_Mean2DXs[sortIdx] = vec4(0);
	gDL_DConics_Mean2DYs[sortIdx] = vec4(0);
	gDL_DViewOpacities[sortIdx] = 0;
}
#endif

// DL_DSplatView AtomicAdd
#ifdef RASTERIZER_ATOMICADD_DL_DSPLAT_VIEW
#extension GL_EXT_shader_atomic_float : require
layout(std430, binding = SBUF_DL_DCOLORS_MEAN2DXS_BINDING) buffer bDL_DColors_Mean2DXs { vec4 gDL_DColors_Mean2DXs[]; };
layout(std430, binding = SBUF_DL_DCONICS_MEAN2DYS_BINDING) buffer bDL_DConics_Mean2DYs { vec4 gDL_DConics_Mean2DYs[]; };
layout(std430, binding = SBUF_DL_DVIEW_OPACITIES_BINDING) buffer bDL_DViewOpacities { float gDL_DViewOpacities[]; };
void atomicAddDL_DSplatView(uint sortIdx, SplatView dL_dSplatView) {
	atomicAdd(gDL_DColors_Mean2DXs[sortIdx].x, dL_dSplatView.color.x);
	atomicAdd(gDL_DColors_Mean2DXs[sortIdx].y, dL_dSplatView.color.y);
	atomicAdd(gDL_DColors_Mean2DXs[sortIdx].z, dL_dSplatView.color.z);
	atomicAdd(gDL_DColors_Mean2DXs[sortIdx].w, dL_dSplatView.geom.mean2D.x);
	atomicAdd(gDL_DConics_Mean2DYs[sortIdx].x, dL_dSplatView.geom.conic.x);
	atomicAdd(gDL_DConics_Mean2DYs[sortIdx].y, dL_dSplatView.geom.conic.y);
	atomicAdd(gDL_DConics_Mean2DYs[sortIdx].z, dL_dSplatView.geom.conic.z);
	atomicAdd(gDL_DConics_Mean2DYs[sortIdx].w, dL_dSplatView.geom.mean2D.y);
	atomicAdd(gDL_DViewOpacities[sortIdx], dL_dSplatView.geom.opacity);
	/* gDL_DColors_Mean2DXs[sortIdx] = vec4(dL_dSplatView.color, dL_dSplatView.geom.mean2D.x);
	gDL_DConics_Mean2DYs[sortIdx] = vec4(dL_dSplatView.geom.conic, dL_dSplatView.geom.mean2D.y);
	gDL_DViewOpacities[sortIdx] = dL_dSplatView.geom.opacity; */
}
#endif

#ifdef RASTERIZER_REDUCE_DL_DSPLAT_VIEW
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_quad : require
// #extension GL_KHR_shader_subgroup_shuffle_relative : require
SplatView zeroDL_DSplatView() {
	SplatView dL_dSplatView;
	dL_dSplatView.color = vec3(0);
	dL_dSplatView.geom.mean2D = vec2(0);
	dL_dSplatView.geom.conic = vec3(0);
	dL_dSplatView.geom.opacity = 0;
	return dL_dSplatView;
}
SplatView subgroupReduceDL_DSplatView(SplatView dL_dSplatView) {
	/* [[unroll]] for (uint delta = 1; delta < gl_SubgroupSize; delta <<= 1) {
	    dL_dSplatView.geom.conic += subgroupShuffleDown(dL_dSplatView.geom.conic, delta);
	    dL_dSplatView.geom.mean2D += subgroupShuffleDown(dL_dSplatView.geom.mean2D, delta);
	    dL_dSplatView.geom.opacity += subgroupShuffleDown(dL_dSplatView.geom.opacity, delta);
	    dL_dSplatView.color += subgroupShuffleDown(dL_dSplatView.color, delta);
	} */
	dL_dSplatView.geom.conic = subgroupAdd(dL_dSplatView.geom.conic);
	dL_dSplatView.geom.mean2D = subgroupAdd(dL_dSplatView.geom.mean2D);
	dL_dSplatView.geom.opacity = subgroupAdd(dL_dSplatView.geom.opacity);
	dL_dSplatView.color = subgroupAdd(dL_dSplatView.color);
	return dL_dSplatView;
}
SplatView quadReduceDL_DSplatView(SplatView dL_dSplatView) {
	dL_dSplatView.geom.conic += subgroupQuadSwapHorizontal(dL_dSplatView.geom.conic);
	dL_dSplatView.geom.mean2D += subgroupQuadSwapHorizontal(dL_dSplatView.geom.mean2D);
	dL_dSplatView.geom.opacity += subgroupQuadSwapHorizontal(dL_dSplatView.geom.opacity);
	dL_dSplatView.color += subgroupQuadSwapHorizontal(dL_dSplatView.color);
	dL_dSplatView.geom.conic += subgroupQuadSwapVertical(dL_dSplatView.geom.conic);
	dL_dSplatView.geom.mean2D += subgroupQuadSwapVertical(dL_dSplatView.geom.mean2D);
	dL_dSplatView.geom.opacity += subgroupQuadSwapVertical(dL_dSplatView.geom.opacity);
	dL_dSplatView.color += subgroupQuadSwapVertical(dL_dSplatView.color);
	return dL_dSplatView;
}
#endif

#ifdef RASTERIZER_VERBOSE
layout(std430, binding = SBUF_VERBOSE_FRAGMENT_COUNT_BINDING) buffer bVerboseFragmentCount {
	uint gVerboseFragmentCount[];
};
layout(std430, binding = SBUF_VERBOSE_COHERENT_FRAGMENT_COUNT_BINDING) buffer bVerboseCoherentFragmentCount {
	uint gVerboseCoherentFragmentCount[];
};
layout(std430, binding = SBUF_VERBOSE_ATOMIC_ADD_COUNT_BINDING) buffer bVerboseAtomicAddCount {
	uint gVerboseAtomicAddCount[];
};
#define VERBOSE_ADD(NAME, COUNT) atomicAdd(gVerbose##NAME[0], COUNT)
#endif

#ifdef RASTERIZER_THREAD_GROUP_TILING_X
/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Divide the 2D-Dispatch_Grid into tiles of dimension [N, DipatchGridDim.y]
// “CTA” (Cooperative Thread Array) == Thread Group in DirectX terminology
struct VGroupThreadID {
	uvec2 groupID;
	uvec2 threadID;
};

// User parameter (N). Recommended values: 8, 16 or 32.
VGroupThreadID ThreadGroupTilingX(uvec2 ctaDim, uint maxTileWidth) {
	const uvec2 dispatchGridDim =
	    gl_NumWorkGroups.xy; // Arguments of the Dispatch call (typically from a ConstantBuffer)
	const uvec2 groupThreadID = gl_LocalInvocationID.xy; // SV_GroupThreadID
	const uvec2 groupId = gl_WorkGroupID.xy;             // SV_GroupID
	// A perfect tile is one with dimensions = [maxTileWidth, dispatchGridDim.y]
	const uint Number_of_CTAs_in_a_perfect_tile = maxTileWidth * dispatchGridDim.y;

	// Possible number of perfect tiles
	const uint Number_of_perfect_tiles = dispatchGridDim.x / maxTileWidth;

	// Total number of CTAs present in the perfect tiles
	const uint Total_CTAs_in_all_perfect_tiles = Number_of_perfect_tiles * maxTileWidth * dispatchGridDim.y;
	const uint vThreadGroupIDFlattened = dispatchGridDim.x * groupId.y + groupId.x;

	// Tile_ID_of_current_CTA : current CTA to TILE-ID mapping.
	const uint Tile_ID_of_current_CTA = vThreadGroupIDFlattened / Number_of_CTAs_in_a_perfect_tile;
	const uint Local_CTA_ID_within_current_tile = vThreadGroupIDFlattened % Number_of_CTAs_in_a_perfect_tile;
	uint Local_CTA_ID_y_within_current_tile;
	uint Local_CTA_ID_x_within_current_tile;

	if (Total_CTAs_in_all_perfect_tiles <= vThreadGroupIDFlattened) {
		// Path taken only if the last tile has imperfect dimensions and CTAs from the last tile are launched.
		uint X_dimension_of_last_tile = dispatchGridDim.x % maxTileWidth;
		Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / X_dimension_of_last_tile;
		Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % X_dimension_of_last_tile;
	} else {
		Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / maxTileWidth;
		Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % maxTileWidth;
	}

	const uint Swizzled_vThreadGroupIDFlattened = Tile_ID_of_current_CTA * maxTileWidth +
	                                              Local_CTA_ID_y_within_current_tile * dispatchGridDim.x +
	                                              Local_CTA_ID_x_within_current_tile;

	uvec2 SwizzledvThreadGroupID;
	SwizzledvThreadGroupID.y = Swizzled_vThreadGroupIDFlattened / dispatchGridDim.x;
	SwizzledvThreadGroupID.x = Swizzled_vThreadGroupIDFlattened % dispatchGridDim.x;

	uvec2 SwizzledvThreadID;
	SwizzledvThreadID.x = ctaDim.x * SwizzledvThreadGroupID.x + groupThreadID.x;
	SwizzledvThreadID.y = ctaDim.y * SwizzledvThreadGroupID.y + groupThreadID.y;

	VGroupThreadID vID;
	vID.groupID = SwizzledvThreadGroupID;
	vID.threadID = SwizzledvThreadID;
	return vID;
}
#endif

#endif
