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
#ifndef RASTERIZER_LOAD_SPLAT
layout(std430, binding = SBUF_OPACITIES_BINDING) readonly buffer bOpacities { float gOpacities[]; };
#endif
layout(std430, binding = SBUF_COLORS_MEAN2DXS_BINDING) readonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = SBUF_CONICS_MEAN2DYS_BINDING) readonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
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
layout(std430, binding = SBUF_QUADS_BINDING) readonly buffer bQuads { vec4 gQuads[]; };
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
layout(std430, binding = SBUF_COLORS_MEAN2DXS_BINDING) writeonly buffer bColors_Mean2DXs { vec4 gColors_Mean2DXs[]; };
layout(std430, binding = SBUF_CONICS_MEAN2DYS_BINDING) writeonly buffer bConics_Mean2DYs { vec4 gConics_Mean2DYs[]; };
void storeSplatView(uint splatIdx, SplatView splatView) {
	gColors_Mean2DXs[splatIdx] = vec4(splatView.color, splatView.geom.mean2D.x);
	gConics_Mean2DYs[splatIdx] = vec4(splatView.geom.conic, splatView.geom.mean2D.y);
}
#endif

// SplatQuad Store
#ifdef RASTERIZER_STORE_SPLAT_QUAD
layout(std430, binding = SBUF_QUADS_BINDING) writeonly buffer bQuads { vec4 gQuads[]; };
void storeSplatQuad(uint splatIdx, SplatQuad quad) { gQuads[splatIdx] = vec4(quad.axis1, quad.axis2); }
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
