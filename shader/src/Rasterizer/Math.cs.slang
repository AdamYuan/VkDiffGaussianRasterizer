#include "Math.slangh"

RWStructuredBuffer<SplatView> gSplatViews;
RWStructuredBuffer<SplatQuad> gSplatQuads;
RWStructuredBuffer<Splat> gSplats;
RWStructuredBuffer<float> gAlphas;

uniform Camera gCamera;

void main() {
	gSplatViews[0] = splat2splatView(gSplats[0], gCamera, gSplatQuads[0]);
	gSplats[0] = bwd_splat2splatView(gSplats[0], gCamera, gSplatViews[0]);
	gAlphas[0] = splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamera);
	gSplatViews[0].geom =
	    bwd_splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamera, gAlphas[0]);
}
