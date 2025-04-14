#version 460

#define RASTERIZER_LOAD_SPLAT_VIEW
#define RASTERIZER_LOAD_SPLAT_QUAD
#include "Common.glsl"

#define VERT_NUM 4
const vec2[VERT_NUM] kVerts = {
    vec2(1.000000, 1.000000),
    vec2(-1.000000, 1.000000),
    vec2(1.000000, -1.000000),
    vec2(-1.000000, -1.000000),
};

out bOut {
	layout(location = 0) flat vec3 color;
	layout(location = 1) flat vec3 conic;
	layout(location = 2) flat vec2 mean2D;
	layout(location = 3) flat float opacity;
	layout(location = 4) flat uint sortIdx;
	layout(location = 5) noperspective vec2 quadPos;
}
gOut;

layout(std430, binding = SBUF_SORT_PAYLOADS_BINDING) readonly buffer bSortPayloads { uint gSortPayloads[]; };

void main() {
	uint orderIdx = gl_InstanceIndex;
	uint sortIdx = gSortPayloads[orderIdx];
	SplatView splatView = loadSplatView(sortIdx);
	SplatQuad splatQuad = loadSplatQuad(sortIdx);

	float quadBound = opacity2quadBound(splatView.geom.opacity);

	Camera camera = loadCamera();
	vec2 meanClip = pos2D2clip(splatView.geom.mean2D, camera);
	vec2 axisClip1 = axis2D2clip(splatQuad.axis1, camera);
	vec2 axisClip2 = axis2D2clip(splatQuad.axis2, camera);

	gOut.color = splatView.color;
	gOut.conic = splatView.geom.conic;
	gOut.mean2D = splatView.geom.mean2D;
	gOut.opacity = splatView.geom.opacity;
	gOut.sortIdx = sortIdx;
	gOut.quadPos = kVerts[gl_VertexIndex] * quadBound;
	gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
}
