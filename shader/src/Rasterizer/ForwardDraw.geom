#version 460

#define RASTERIZER_LOAD_SPLAT_VIEW
#define RASTERIZER_LOAD_SPLAT_QUAD
#include "Common.glsl"

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

in bIn { layout(location = 0) uint instanceID; }
gIn[];

out bOut {
	layout(location = 0) flat float opacity;
	layout(location = 1) flat vec3 color;
	layout(location = 2) noperspective vec2 quadPos;
}
gOut;

layout(std430, binding = B_SORT_PAYLOADS_BINDING) readonly buffer bSortPayloads { uint gSortPayloads[]; };

void main() {
	uint splatIdx = gSortPayloads[gIn[0].instanceID];
	SplatView splatView = loadSplatView(splatIdx);
	SplatQuad splatQuad = loadSplatQuad(splatIdx);

	float quadBound = opacity2quadBound(splatView.geom.opacity);

	Camera camera = loadCamera();
	vec2 meanClip = pos2D2clip(splatView.geom.mean2D, camera);
	vec2 axisClip1 = axis2D2clip(splatQuad.axis1, camera);
	vec2 axisClip2 = axis2D2clip(splatQuad.axis2, camera);

	gOut.opacity = splatView.geom.opacity;
	gOut.color = splatView.color;
	gOut.quadPos = vec2(-quadBound, -quadBound);
	gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
	EmitVertex();

	gOut.opacity = splatView.geom.opacity;
	gOut.color = splatView.color;
	gOut.quadPos = vec2(quadBound, -quadBound);
	gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
	EmitVertex();

	gOut.opacity = splatView.geom.opacity;
	gOut.color = splatView.color;
	gOut.quadPos = vec2(-quadBound, quadBound);
	gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
	EmitVertex();

	gOut.opacity = splatView.geom.opacity;
	gOut.color = splatView.color;
	gOut.quadPos = vec2(quadBound, quadBound);
	gl_Position = vec4(meanClip + axisClip1 * gOut.quadPos.x + axisClip2 * gOut.quadPos.y, 0, 1);
	EmitVertex();

	EndPrimitive();
}
