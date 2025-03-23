#include "Math.slangh"

RWStructuredBuffer<SplatView> gSplatViews;
RWStructuredBuffer<SplatQuad> gSplatQuads;
RWStructuredBuffer<Splat> gSplats;
RWStructuredBuffer<float> gAlphas;

uniform float3 gCamPos;
uniform float3x3 gCamViewMat;
uniform float2 gCamFocal, gCamResolution;

void main() {
	gSplatViews[0] = splat2splatView(gSplats[0], gCamPos, gCamViewMat, gCamFocal, gCamResolution, gSplatQuads[0]);
	gSplats[0] = bwd_splat2splatView(gSplats[0], gCamPos, gCamViewMat, gCamFocal, gCamResolution, gSplatViews[0]);
	gAlphas[0] = splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamResolution.y);
	gSplatViews[0].geom =
	    bwd_splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamResolution.y, gAlphas[0]);
}
