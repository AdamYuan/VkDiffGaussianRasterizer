#version 460

#define RASTERIZER_LOAD_SPLAT_VIEW
#define RASTERIZER_LOAD_SPLAT_QUAD
#include "Common.glsl"

// #define DEBUG_SUBGROUP

#define VERT_NUM 5
const vec2[VERT_NUM] kVerts = {
    vec2(1.000000, 0.726543),  vec2(-0.381966, 1.175570),  vec2(1.000000, -0.726543),
    vec2(-1.236068, 0.000000), vec2(-0.381966, -1.175570),
};

layout(points) in;
layout(triangle_strip, max_vertices = VERT_NUM) out;

in bIn { layout(location = 0) uint instanceID; }
gIn[];

out bOut {
	layout(location = 0) flat float opacity;
	layout(location = 1) flat vec3 color;
	layout(location = 2) noperspective vec2 quadPos;
#ifdef DEBUG_SUBGROUP
	layout(location = 3) flat uint sortIdx;
#endif
}
gOut;

layout(std430, binding = SBUF_SORT_PAYLOADS_BINDING) readonly buffer bSortPayloads { uint gSortPayloads[]; };

void main() {
	uint sortIdx = gSortPayloads[gIn[0].instanceID];
	SplatView splatView = loadSplatView(sortIdx);
	SplatQuad splatQuad = loadSplatQuad(sortIdx);

	float quadBound = opacity2quadBound(splatView.geom.opacity);

	Camera camera = loadCamera();
	vec2 meanClip = pos2D2clip(splatView.geom.mean2D, camera);
	vec2 axisClip1 = axis2D2clip(splatQuad.axis1, camera);
	vec2 axisClip2 = axis2D2clip(splatQuad.axis2, camera);

	[[unroll]]
	for (uint i = 0; i < VERT_NUM; ++i) {
		gOut.opacity = splatView.geom.opacity;
		gOut.color = splatView.color;
		gOut.quadPos = kVerts[i] * quadBound;
#ifdef DEBUG_SUBGROUP
		gOut.sortIdx = sortIdx;
#endif
		gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
		EmitVertex();
	}

	EndPrimitive();
}
