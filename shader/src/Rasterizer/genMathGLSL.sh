#!/bin/bash
~/.cache/packman/chk/slang/2025.4/bin/slangc Math.cs.slang -stage compute -target glsl -o Math.glsl
sed -i '/^#/d' Math.glsl # Remove all lines starting with '#'
sed -i '/^$/d' Math.glsl # Remove all empty lines
sed -i -n '/std430/,/^}/!p' Math.glsl # Remove all std430 blocks
sed -i -n '/std140/,/^}/!p' Math.glsl # Remove all std140 blocks
sed -i '/layout/d' Math.glsl # Remove all lines containing 'layout'
sed -i '/void main()/,$d' Math.glsl # Remove main() function

# Remove suffix in struct and function names
keywords=(
	"Camera"
	"pos"
	"focal"
	"viewMat"
	"resolution"
	"SH" 
	"data" 
	"SplatGeom" 
	"mean" 
	"scale" 
	"quat" 
	"opacity" 
	"Splat" 
	"geom"
	"sh"
	"SplatViewGeom"
	"mean2D"
	"conic"
	"SplatView"
	"color"
	"SplatQuad"
	"axis1"
	"axis2"
	"splat2splatView"
	"bwd_splat2splatView"
	"splatViewGeom2alpha"
	"bwd_splatViewGeom2alpha"
	"behindFrustum"
	"inFrustum"
)
for keyword in "${keywords[@]}"; do
    sed -i "s/\\<${keyword}_[0-9]\\>/${keyword}/g" Math.glsl
done

sed -i '1i#define RASTERIZER_MATH_GLSL' Math.glsl
sed -i '1i#ifndef RASTERIZER_MATH_GLSL' Math.glsl

sed -i '$a\#endif' Math.glsl

# Clang Format
clang-format -i Math.glsl
