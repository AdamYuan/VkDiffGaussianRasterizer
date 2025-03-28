#version 460
#extension GL_ARB_fragment_shader_interlock : require

#include "Common.glsl"

in bIn {
	layout(location = 0) flat float opacity;
	layout(location = 1) flat vec3 color;
	layout(location = 2) noperspective vec2 quadPos;
}
gIn;

layout(rgba32f, binding = I_COLOR0_BINDING) coherent uniform image2D gColorT;

layout(pixel_interlock_ordered) in;

void main() {
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity);
	if (alpha < ALPHA_MIN)
		discard;

	alpha = min(alpha, ALPHA_MAX);
	float oneMinusAlpha = 1.0 - alpha;
	vec3 alphaColor = alpha * gIn.color;

	ivec2 coord = ivec2(gl_FragCoord.xy);

	beginInvocationInterlockARB();
	vec4 color_T = imageLoad(gColorT, coord);
	color_T *= oneMinusAlpha;
	color_T.xyz += alphaColor;
	imageStore(gColorT, coord, color_T);
	endInvocationInterlockARB();
}
