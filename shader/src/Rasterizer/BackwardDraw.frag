#version 460
#extension GL_ARB_fragment_shader_interlock : require

#include "Common.glsl"

in bIn {
	layout(location = 0) flat vec4 color_mean2Dx;
	layout(location = 1) flat vec4 conic_mean2Dy;
	layout(location = 2) flat float opacity;
	layout(location = 3) noperspective vec2 quadPos;
}
gIn;

layout(rgba32f, binding = SIMG_IMAGE0_BINDING) coherent uniform image2D gColors_Ts;

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
	vec4 color_T = imageLoad(gColors_Ts, coord);
	color_T *= oneMinusAlpha;
	color_T.xyz += alphaColor;
	imageStore(gColors_Ts, coord, color_T);
	endInvocationInterlockARB();

	SplatView splatView;
	splatView.color = gIn.color_mean2Dx.xyz;
	splatView.geom.conic = gIn.conic_mean2Dy.xyz;
	splatView.geom.mean2D = vec2(gIn.color_mean2Dx.w, gIn.conic_mean2Dy.w);
	splatView.geom.opacity = gIn.opacity;
}
