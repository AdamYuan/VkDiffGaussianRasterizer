#version 460

#include "Common.glsl"

in bIn {
	layout(location = 0) flat float opacity;
	layout(location = 1) flat vec3 color;
	layout(location = 2) noperspective vec2 quadPos;
}
gIn;

layout(location = 0) out vec4 gOutFragColor;

void main() {
	float alpha = quadPos2alpha(gIn.quadPos, gIn.opacity);
	if (alpha < ALPHA_MIN)
		discard;
	alpha = min(alpha, ALPHA_MAX);
	gOutFragColor = vec4(gIn.color, alpha);
}
