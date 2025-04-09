#include "Math.slangh"

RWStructuredBuffer<SplatView> gSplatViews;
RWStructuredBuffer<SplatQuad> gSplatQuads;
RWStructuredBuffer<Splat> gSplats;
RWStructuredBuffer<float> gAlphas;
RWStructuredBuffer<float2> g2Ds;
RWStructuredBuffer<bool> gBools;

uniform Camera gCamera;

void main() {
	gSplatViews[0] = splat2splatView(gSplats[0], gCamera, gSplatQuads[0]);
	gSplats[0] = bwd_splat2splatView(gSplats[0], gCamera, gSplatViews[0]);
	float G = 2.0;
	gAlphas[0] = splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamera);
	gBools[0] = behindFrustum(gSplats[0], gCamera, gAlphas[10]);
	gBools[1] = inFrustum(gSplatViews[0].geom, gSplatQuads[0], gCamera);
	gAlphas[3] = opacity2quadBound(gAlphas[23]);
	g2Ds[0] = pos2D2clip(g2Ds[1], gCamera);
	g2Ds[3] = axis2D2clip(g2Ds[4], gCamera);
	gAlphas[35] = quadPos2alpha(g2Ds[12], gAlphas[23]);
	gAlphas[22] = quadPos2alpha(g2Ds[12], gAlphas[23], G);
	gSplatViews[0].geom =
	    bwd_splatViewGeom2alpha(gSplatViews[0].geom, float2(gAlphas[1], gAlphas[2]), gCamera, G, gAlphas[0]);
}
