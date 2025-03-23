#ifndef RASTERIZER_MATH_GLSL
#define RASTERIZER_MATH_GLSL
struct SplatViewGeom {
	vec3 conic;
	vec2 mean2D;
	float opacity;
};
struct SplatView {
	SplatViewGeom geom;
	vec3 color;
};
struct SplatGeom {
	vec4 quat;
	vec3 scale;
	vec3 mean;
	float opacity;
};
struct SH {
	vec3 data[16];
};
struct Splat {
	SplatGeom geom;
	SH sh;
};
struct SplatQuad {
	vec2 axis1;
	vec2 axis2;
};
struct Camera {
	vec3 pos;
	vec2 focal;
	mat3x3 viewMat;
	uvec2 resolution;
};
struct DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 {
	mat3x3 primal_0;
	mat3x3 differential_0;
};
struct DiffPair_vectorx3Cfloatx2C3x3E_0 {
	vec3 primal_0;
	vec3 differential_0;
};
void _d_mul_0(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 left_0, inout DiffPair_vectorx3Cfloatx2C3x3E_0 right_0,
              vec3 dOut_0) {
	vec3 right_d_result_0;
	mat3x3 left_d_result_0;
	float _S34 = left_0.primal_0[0][0] * dOut_0[0];
	left_d_result_0[0][0] = right_0.primal_0[0] * dOut_0[0];
	float sum_0 = _S34 + left_0.primal_0[1][0] * dOut_0[1];
	left_d_result_0[1][0] = right_0.primal_0[0] * dOut_0[1];
	float sum_1 = sum_0 + left_0.primal_0[2][0] * dOut_0[2];
	left_d_result_0[2][0] = right_0.primal_0[0] * dOut_0[2];
	right_d_result_0[0] = sum_1;
	float _S35 = left_0.primal_0[0][1] * dOut_0[0];
	left_d_result_0[0][1] = right_0.primal_0[1] * dOut_0[0];
	float sum_2 = _S35 + left_0.primal_0[1][1] * dOut_0[1];
	left_d_result_0[1][1] = right_0.primal_0[1] * dOut_0[1];
	float sum_3 = sum_2 + left_0.primal_0[2][1] * dOut_0[2];
	left_d_result_0[2][1] = right_0.primal_0[1] * dOut_0[2];
	right_d_result_0[1] = sum_3;
	float _S36 = left_0.primal_0[0][2] * dOut_0[0];
	left_d_result_0[0][2] = right_0.primal_0[2] * dOut_0[0];
	float sum_4 = _S36 + left_0.primal_0[1][2] * dOut_0[1];
	left_d_result_0[1][2] = right_0.primal_0[2] * dOut_0[1];
	float sum_5 = sum_4 + left_0.primal_0[2][2] * dOut_0[2];
	left_d_result_0[2][2] = right_0.primal_0[2] * dOut_0[2];
	right_d_result_0[2] = sum_5;
	left_0.primal_0 = left_0.primal_0;
	left_0.differential_0 = left_d_result_0;
	right_0.primal_0 = right_0.primal_0;
	right_0.differential_0 = right_d_result_0;
	return;
}
struct DiffPair_float_0 {
	float primal_0;
	float differential_0;
};
void _d_max_0(inout DiffPair_float_0 dpx_0, inout DiffPair_float_0 dpy_0, float dOut_1) {
	DiffPair_float_0 _S37 = dpx_0;
	float _S38;
	if ((dpx_0.primal_0) > (dpy_0.primal_0)) {
		_S38 = dOut_1;
	} else {
		_S38 = 0.0;
	}
	dpx_0.primal_0 = _S37.primal_0;
	dpx_0.differential_0 = _S38;
	DiffPair_float_0 _S39 = dpy_0;
	if ((dpy_0.primal_0) > (_S37.primal_0)) {
		_S38 = dOut_1;
	} else {
		_S38 = 0.0;
	}
	dpy_0.primal_0 = _S39.primal_0;
	dpy_0.differential_0 = _S38;
	return;
}
void _d_max_vector_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_1, inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpy_1,
                     vec3 dOut_2) {
	vec3 left_d_result_1;
	vec3 right_d_result_1;
	DiffPair_float_0 left_dp_0;
	left_dp_0.primal_0 = dpx_1.primal_0[0];
	left_dp_0.differential_0 = 0.0;
	DiffPair_float_0 right_dp_0;
	right_dp_0.primal_0 = dpy_1.primal_0[0];
	right_dp_0.differential_0 = 0.0;
	_d_max_0(left_dp_0, right_dp_0, dOut_2[0]);
	left_d_result_1[0] = left_dp_0.differential_0;
	right_d_result_1[0] = right_dp_0.differential_0;
	DiffPair_float_0 left_dp_1;
	left_dp_1.primal_0 = dpx_1.primal_0[1];
	left_dp_1.differential_0 = 0.0;
	DiffPair_float_0 right_dp_1;
	right_dp_1.primal_0 = dpy_1.primal_0[1];
	right_dp_1.differential_0 = 0.0;
	_d_max_0(left_dp_1, right_dp_1, dOut_2[1]);
	left_d_result_1[1] = left_dp_1.differential_0;
	right_d_result_1[1] = right_dp_1.differential_0;
	DiffPair_float_0 left_dp_2;
	left_dp_2.primal_0 = dpx_1.primal_0[2];
	left_dp_2.differential_0 = 0.0;
	DiffPair_float_0 right_dp_2;
	right_dp_2.primal_0 = dpy_1.primal_0[2];
	right_dp_2.differential_0 = 0.0;
	_d_max_0(left_dp_2, right_dp_2, dOut_2[2]);
	left_d_result_1[2] = left_dp_2.differential_0;
	right_d_result_1[2] = right_dp_2.differential_0;
	dpx_1.primal_0 = dpx_1.primal_0;
	dpx_1.differential_0 = left_d_result_1;
	dpy_1.primal_0 = dpy_1.primal_0;
	dpy_1.differential_0 = right_d_result_1;
	return;
}
void _d_clamp_0(inout DiffPair_float_0 dpx_2, inout DiffPair_float_0 dpMin_0, inout DiffPair_float_0 dpMax_0,
                float dOut_3) {
	DiffPair_float_0 _S40 = dpx_2;
	bool _S41;
	if ((dpx_2.primal_0) > (dpMin_0.primal_0)) {
		_S41 = (dpx_2.primal_0) < (dpMax_0.primal_0);
	} else {
		_S41 = false;
	}
	float _S42;
	if (_S41) {
		_S42 = dOut_3;
	} else {
		_S42 = 0.0;
	}
	dpx_2.primal_0 = _S40.primal_0;
	dpx_2.differential_0 = _S42;
	DiffPair_float_0 _S43 = dpMin_0;
	if ((_S40.primal_0) <= (dpMin_0.primal_0)) {
		_S42 = dOut_3;
	} else {
		_S42 = 0.0;
	}
	dpMin_0.primal_0 = _S43.primal_0;
	dpMin_0.differential_0 = _S42;
	DiffPair_float_0 _S44 = dpMax_0;
	if ((dpx_2.primal_0) >= (dpMax_0.primal_0)) {
		_S42 = dOut_3;
	} else {
		_S42 = 0.0;
	}
	dpMax_0.primal_0 = _S44.primal_0;
	dpMax_0.differential_0 = _S42;
	return;
}
struct DiffPair_vectorx3Cfloatx2C2x3E_0 {
	vec2 primal_0;
	vec2 differential_0;
};
void _d_clamp_vector_0(inout DiffPair_vectorx3Cfloatx2C2x3E_0 dpx_3, inout DiffPair_vectorx3Cfloatx2C2x3E_0 dpy_2,
                       inout DiffPair_vectorx3Cfloatx2C2x3E_0 dpz_0, vec2 dOut_4) {
	vec2 left_d_result_2;
	vec2 middle_d_result_0;
	vec2 right_d_result_2;
	DiffPair_float_0 left_dp_3;
	left_dp_3.primal_0 = dpx_3.primal_0[0];
	left_dp_3.differential_0 = 0.0;
	DiffPair_float_0 middle_dp_0;
	middle_dp_0.primal_0 = dpy_2.primal_0[0];
	middle_dp_0.differential_0 = 0.0;
	DiffPair_float_0 right_dp_3;
	right_dp_3.primal_0 = dpz_0.primal_0[0];
	right_dp_3.differential_0 = 0.0;
	_d_clamp_0(left_dp_3, middle_dp_0, right_dp_3, dOut_4[0]);
	left_d_result_2[0] = left_dp_3.differential_0;
	middle_d_result_0[0] = middle_dp_0.differential_0;
	right_d_result_2[0] = right_dp_3.differential_0;
	DiffPair_float_0 left_dp_4;
	left_dp_4.primal_0 = dpx_3.primal_0[1];
	left_dp_4.differential_0 = 0.0;
	DiffPair_float_0 middle_dp_1;
	middle_dp_1.primal_0 = dpy_2.primal_0[1];
	middle_dp_1.differential_0 = 0.0;
	DiffPair_float_0 right_dp_4;
	right_dp_4.primal_0 = dpz_0.primal_0[1];
	right_dp_4.differential_0 = 0.0;
	_d_clamp_0(left_dp_4, middle_dp_1, right_dp_4, dOut_4[1]);
	left_d_result_2[1] = left_dp_4.differential_0;
	middle_d_result_0[1] = middle_dp_1.differential_0;
	right_d_result_2[1] = right_dp_4.differential_0;
	dpx_3.primal_0 = dpx_3.primal_0;
	dpx_3.differential_0 = left_d_result_2;
	dpy_2.primal_0 = dpy_2.primal_0;
	dpy_2.differential_0 = middle_d_result_0;
	dpz_0.primal_0 = dpz_0.primal_0;
	dpz_0.differential_0 = right_d_result_2;
	return;
}
mat3x3 scale2matrix_0(vec3 scale) { return mat3x3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z); }
mat3x3 quat2matrix_0(vec4 quat) {
	float x_0 = quat.x;
	float y_0 = quat.y;
	float z_0 = quat.z;
	float w_0 = quat.w;
	float _S45 = y_0 * y_0;
	float _S46 = z_0 * z_0;
	float _S47 = x_0 * y_0;
	float _S48 = w_0 * z_0;
	float _S49 = x_0 * z_0;
	float _S50 = w_0 * y_0;
	float _S51 = x_0 * x_0;
	float _S52 = y_0 * z_0;
	float _S53 = w_0 * x_0;
	return mat3x3(vec3(1.0 - 2.0 * (_S45 + _S46), 2.0 * (_S47 - _S48), 2.0 * (_S49 + _S50)),
	              vec3(2.0 * (_S47 + _S48), 1.0 - 2.0 * (_S51 + _S46), 2.0 * (_S52 - _S53)),
	              vec3(2.0 * (_S49 - _S50), 2.0 * (_S52 + _S53), 1.0 - 2.0 * (_S51 + _S45)));
}
void mul_0(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 left_1, inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 right_1,
           mat3x3 dOut_5) {
	mat3x3 left_d_result_3;
	left_d_result_3[0][0] = 0.0;
	left_d_result_3[0][1] = 0.0;
	left_d_result_3[0][2] = 0.0;
	left_d_result_3[1][0] = 0.0;
	left_d_result_3[1][1] = 0.0;
	left_d_result_3[1][2] = 0.0;
	left_d_result_3[2][0] = 0.0;
	left_d_result_3[2][1] = 0.0;
	left_d_result_3[2][2] = 0.0;
	mat3x3 right_d_result_3;
	right_d_result_3[0][0] = 0.0;
	right_d_result_3[0][1] = 0.0;
	right_d_result_3[0][2] = 0.0;
	right_d_result_3[1][0] = 0.0;
	right_d_result_3[1][1] = 0.0;
	right_d_result_3[1][2] = 0.0;
	right_d_result_3[2][0] = 0.0;
	right_d_result_3[2][1] = 0.0;
	right_d_result_3[2][2] = 0.0;
	left_d_result_3[0][0] = left_d_result_3[0][0] + right_1.primal_0[0][0] * dOut_5[0][0];
	right_d_result_3[0][0] = right_d_result_3[0][0] + left_1.primal_0[0][0] * dOut_5[0][0];
	left_d_result_3[0][1] = left_d_result_3[0][1] + right_1.primal_0[1][0] * dOut_5[0][0];
	right_d_result_3[1][0] = right_d_result_3[1][0] + left_1.primal_0[0][1] * dOut_5[0][0];
	left_d_result_3[0][2] = left_d_result_3[0][2] + right_1.primal_0[2][0] * dOut_5[0][0];
	right_d_result_3[2][0] = right_d_result_3[2][0] + left_1.primal_0[0][2] * dOut_5[0][0];
	left_d_result_3[0][0] = left_d_result_3[0][0] + right_1.primal_0[0][1] * dOut_5[0][1];
	right_d_result_3[0][1] = right_d_result_3[0][1] + left_1.primal_0[0][0] * dOut_5[0][1];
	left_d_result_3[0][1] = left_d_result_3[0][1] + right_1.primal_0[1][1] * dOut_5[0][1];
	right_d_result_3[1][1] = right_d_result_3[1][1] + left_1.primal_0[0][1] * dOut_5[0][1];
	left_d_result_3[0][2] = left_d_result_3[0][2] + right_1.primal_0[2][1] * dOut_5[0][1];
	right_d_result_3[2][1] = right_d_result_3[2][1] + left_1.primal_0[0][2] * dOut_5[0][1];
	left_d_result_3[0][0] = left_d_result_3[0][0] + right_1.primal_0[0][2] * dOut_5[0][2];
	right_d_result_3[0][2] = right_d_result_3[0][2] + left_1.primal_0[0][0] * dOut_5[0][2];
	left_d_result_3[0][1] = left_d_result_3[0][1] + right_1.primal_0[1][2] * dOut_5[0][2];
	right_d_result_3[1][2] = right_d_result_3[1][2] + left_1.primal_0[0][1] * dOut_5[0][2];
	left_d_result_3[0][2] = left_d_result_3[0][2] + right_1.primal_0[2][2] * dOut_5[0][2];
	right_d_result_3[2][2] = right_d_result_3[2][2] + left_1.primal_0[0][2] * dOut_5[0][2];
	left_d_result_3[1][0] = left_d_result_3[1][0] + right_1.primal_0[0][0] * dOut_5[1][0];
	right_d_result_3[0][0] = right_d_result_3[0][0] + left_1.primal_0[1][0] * dOut_5[1][0];
	left_d_result_3[1][1] = left_d_result_3[1][1] + right_1.primal_0[1][0] * dOut_5[1][0];
	right_d_result_3[1][0] = right_d_result_3[1][0] + left_1.primal_0[1][1] * dOut_5[1][0];
	left_d_result_3[1][2] = left_d_result_3[1][2] + right_1.primal_0[2][0] * dOut_5[1][0];
	right_d_result_3[2][0] = right_d_result_3[2][0] + left_1.primal_0[1][2] * dOut_5[1][0];
	left_d_result_3[1][0] = left_d_result_3[1][0] + right_1.primal_0[0][1] * dOut_5[1][1];
	right_d_result_3[0][1] = right_d_result_3[0][1] + left_1.primal_0[1][0] * dOut_5[1][1];
	left_d_result_3[1][1] = left_d_result_3[1][1] + right_1.primal_0[1][1] * dOut_5[1][1];
	right_d_result_3[1][1] = right_d_result_3[1][1] + left_1.primal_0[1][1] * dOut_5[1][1];
	left_d_result_3[1][2] = left_d_result_3[1][2] + right_1.primal_0[2][1] * dOut_5[1][1];
	right_d_result_3[2][1] = right_d_result_3[2][1] + left_1.primal_0[1][2] * dOut_5[1][1];
	left_d_result_3[1][0] = left_d_result_3[1][0] + right_1.primal_0[0][2] * dOut_5[1][2];
	right_d_result_3[0][2] = right_d_result_3[0][2] + left_1.primal_0[1][0] * dOut_5[1][2];
	left_d_result_3[1][1] = left_d_result_3[1][1] + right_1.primal_0[1][2] * dOut_5[1][2];
	right_d_result_3[1][2] = right_d_result_3[1][2] + left_1.primal_0[1][1] * dOut_5[1][2];
	left_d_result_3[1][2] = left_d_result_3[1][2] + right_1.primal_0[2][2] * dOut_5[1][2];
	right_d_result_3[2][2] = right_d_result_3[2][2] + left_1.primal_0[1][2] * dOut_5[1][2];
	left_d_result_3[2][0] = left_d_result_3[2][0] + right_1.primal_0[0][0] * dOut_5[2][0];
	right_d_result_3[0][0] = right_d_result_3[0][0] + left_1.primal_0[2][0] * dOut_5[2][0];
	left_d_result_3[2][1] = left_d_result_3[2][1] + right_1.primal_0[1][0] * dOut_5[2][0];
	right_d_result_3[1][0] = right_d_result_3[1][0] + left_1.primal_0[2][1] * dOut_5[2][0];
	left_d_result_3[2][2] = left_d_result_3[2][2] + right_1.primal_0[2][0] * dOut_5[2][0];
	right_d_result_3[2][0] = right_d_result_3[2][0] + left_1.primal_0[2][2] * dOut_5[2][0];
	left_d_result_3[2][0] = left_d_result_3[2][0] + right_1.primal_0[0][1] * dOut_5[2][1];
	right_d_result_3[0][1] = right_d_result_3[0][1] + left_1.primal_0[2][0] * dOut_5[2][1];
	left_d_result_3[2][1] = left_d_result_3[2][1] + right_1.primal_0[1][1] * dOut_5[2][1];
	right_d_result_3[1][1] = right_d_result_3[1][1] + left_1.primal_0[2][1] * dOut_5[2][1];
	left_d_result_3[2][2] = left_d_result_3[2][2] + right_1.primal_0[2][1] * dOut_5[2][1];
	right_d_result_3[2][1] = right_d_result_3[2][1] + left_1.primal_0[2][2] * dOut_5[2][1];
	left_d_result_3[2][0] = left_d_result_3[2][0] + right_1.primal_0[0][2] * dOut_5[2][2];
	right_d_result_3[0][2] = right_d_result_3[0][2] + left_1.primal_0[2][0] * dOut_5[2][2];
	left_d_result_3[2][1] = left_d_result_3[2][1] + right_1.primal_0[1][2] * dOut_5[2][2];
	right_d_result_3[1][2] = right_d_result_3[1][2] + left_1.primal_0[2][1] * dOut_5[2][2];
	left_d_result_3[2][2] = left_d_result_3[2][2] + right_1.primal_0[2][2] * dOut_5[2][2];
	right_d_result_3[2][2] = right_d_result_3[2][2] + left_1.primal_0[2][2] * dOut_5[2][2];
	left_1.primal_0 = left_1.primal_0;
	left_1.differential_0 = left_d_result_3;
	right_1.primal_0 = right_1.primal_0;
	right_1.differential_0 = right_d_result_3;
	return;
}
vec3 cov2conic_0(vec3 cov_0) {
	float _S54 = cov_0.x;
	float _S55 = cov_0.z;
	float _S56 = cov_0.y;
	return vec3(_S55, -_S56, _S54) / (_S54 * _S55 - _S56 * _S56);
}
void _d_sqrt_0(inout DiffPair_float_0 dpx_4, float dOut_6) {
	float _S57 = 0.5 / sqrt(max(1.00000001168609742e-07, dpx_4.primal_0)) * dOut_6;
	dpx_4.primal_0 = dpx_4.primal_0;
	dpx_4.differential_0 = _S57;
	return;
}
vec3 sh2color_0(SH sh, vec3 dir_0) {
	float x_1 = dir_0.x;
	float y_1 = dir_0.y;
	float z_1 = dir_0.z;
	float xx_0 = x_1 * x_1;
	float yy_0 = y_1 * y_1;
	float zz_0 = z_1 * z_1;
	float xy_0 = x_1 * y_1;
	float _S58 = 2.0 * zz_0;
	float _S59 = xx_0 - yy_0;
	float _S60 = 3.0 * xx_0;
	float _S61 = 4.0 * zz_0 - xx_0 - yy_0;
	float _S62 = 3.0 * yy_0;
	return max(0.282094806432724 * sh.data[0] + 0.5 +
	               (-0.48860251903533936 * y_1 * sh.data[1] + 0.48860251903533936 * z_1 * sh.data[2] -
	                0.48860251903533936 * x_1 * sh.data[3]) +
	               (1.09254848957061768 * xy_0 * sh.data[4] + -1.09254848957061768 * (y_1 * z_1) * sh.data[5] +
	                0.31539157032966614 * (_S58 - xx_0 - yy_0) * sh.data[6] +
	                -1.09254848957061768 * (x_1 * z_1) * sh.data[7] + 0.54627424478530884 * _S59 * sh.data[8]) +
	               (-0.59004360437393188 * y_1 * (_S60 - yy_0) * sh.data[9] +
	                2.89061141014099121 * xy_0 * z_1 * sh.data[10] + -0.4570457935333252 * y_1 * _S61 * sh.data[11] +
	                0.37317633628845215 * z_1 * (_S58 - _S60 - _S62) * sh.data[12] +
	                -0.4570457935333252 * x_1 * _S61 * sh.data[13] + 1.44530570507049561 * z_1 * _S59 * sh.data[14] +
	                -0.59004360437393188 * x_1 * (xx_0 - _S62) * sh.data[15]),
	           vec3(0.0));
}
SplatQuad cov2quad_0(vec3 cov_1) {
	float a_0 = cov_1.x;
	float b_0 = cov_1.y;
	float c_0 = cov_1.z;
	float mid_0 = 0.5 * (a_0 + c_0);
	float radius_0 = length(vec2((a_0 - c_0) * 0.5, b_0));
	float lambda1_0 = mid_0 + radius_0;
	float lambda2_0 = mid_0 - radius_0;
	vec2 eigen1_0 = normalize(vec2(b_0, lambda1_0 - a_0));
	vec2 eigen2_0 = vec2(eigen1_0.y, -eigen1_0.x);
	SplatQuad quad_0;
	quad_0.axis1 = eigen1_0 * sqrt(lambda1_0);
	quad_0.axis2 = eigen2_0 * sqrt(lambda2_0);
	return quad_0;
}
SplatView splat2splatView(Splat splat_0, Camera camera_0, out SplatQuad o_splatQuad_0) {
	vec3 camMean_0 = splat_0.geom.mean - camera_0.pos;
	vec3 viewMean_0 = (((camMean_0) * (camera_0.viewMat)));
	float invViewMeanZ_0 = 1.0 / viewMean_0.z;
	vec2 projMean_0 = viewMean_0.xy * camera_0.focal * invViewMeanZ_0;
	vec2 camHalfRes_0 = vec2(camera_0.resolution) * 0.5;
	vec2 clampedProjMean_0 = clamp(projMean_0, -1.29999995231628418 * camHalfRes_0, 1.29999995231628418 * camHalfRes_0);
	mat3x3 JWRS_0 = ((((((scale2matrix_0(splat_0.geom.scale)) * (quat2matrix_0(splat_0.geom.quat))))) *
	                  ((((camera_0.viewMat) *
	                     (mat3x3(vec3(camera_0.focal.x * invViewMeanZ_0, 0.0, -clampedProjMean_0.x * invViewMeanZ_0),
	                             vec3(0.0, camera_0.focal.y * invViewMeanZ_0, -clampedProjMean_0.y * invViewMeanZ_0),
	                             vec3(0.0, 0.0, 0.0))))))));
	mat3x3 _S63 = (((transpose(JWRS_0)) * (JWRS_0)));
	mat2x2 _S64 = mat2x2(_S63[0].xy, _S63[1].xy);
	vec3 cov2D_0 = vec3(_S64[0][0], _S64[0][1], _S64[1][1]) + vec3(0.30000001192092896, 0.0, 0.30000001192092896);
	SplatView splatView_0;
	splatView_0.geom.mean2D = projMean_0 + camHalfRes_0;
	splatView_0.geom.conic = cov2conic_0(cov2D_0);
	splatView_0.geom.opacity = splat_0.geom.opacity;
	splatView_0.color = sh2color_0(splat_0.sh, normalize(camMean_0));
	o_splatQuad_0 = cov2quad_0(cov2D_0);
	return splatView_0;
}
SplatGeom SplatGeom_x24_syn_dzero_0() {
	SplatGeom result_0;
	result_0.quat = vec4(0.0);
	const vec3 _S65 = vec3(0.0);
	result_0.scale = _S65;
	result_0.mean = _S65;
	result_0.opacity = 0.0;
	return result_0;
}
SH SH_x24_syn_dzero_0() {
	SH result_1;
	const vec3 _S66 = vec3(0.0);
	result_1.data[0] = _S66;
	result_1.data[1] = _S66;
	result_1.data[2] = _S66;
	result_1.data[3] = _S66;
	result_1.data[4] = _S66;
	result_1.data[5] = _S66;
	result_1.data[6] = _S66;
	result_1.data[7] = _S66;
	result_1.data[8] = _S66;
	result_1.data[9] = _S66;
	result_1.data[10] = _S66;
	result_1.data[11] = _S66;
	result_1.data[12] = _S66;
	result_1.data[13] = _S66;
	result_1.data[14] = _S66;
	result_1.data[15] = _S66;
	return result_1;
}
Splat Splat_x24_syn_dzero_0() {
	Splat result_2;
	result_2.geom = SplatGeom_x24_syn_dzero_0();
	result_2.sh = SH_x24_syn_dzero_0();
	return result_2;
}
struct DiffPair_Splat_0 {
	Splat primal_0;
	Splat differential_0;
};
vec3 s_primal_ctx_mul_0(mat3x3 _S67, vec3 _S68) { return (((_S68) * (_S67))); }
vec2 s_primal_ctx_clamp_0(vec2 _S69, vec2 _S70, vec2 _S71) { return clamp(_S69, _S70, _S71); }
mat3x3 s_primal_ctx_scale2matrix_0(vec3 dpscale_0) {
	return mat3x3(dpscale_0.x, 0.0, 0.0, 0.0, dpscale_0.y, 0.0, 0.0, 0.0, dpscale_0.z);
}
mat3x3 s_primal_ctx_quat2matrix_0(vec4 dpquat_0) {
	float x_2 = dpquat_0.x;
	float y_2 = dpquat_0.y;
	float z_2 = dpquat_0.z;
	float w_1 = dpquat_0.w;
	float _S72 = y_2 * y_2;
	float _S73 = z_2 * z_2;
	float _S74 = x_2 * y_2;
	float _S75 = w_1 * z_2;
	float _S76 = x_2 * z_2;
	float _S77 = w_1 * y_2;
	float _S78 = x_2 * x_2;
	float _S79 = y_2 * z_2;
	float _S80 = w_1 * x_2;
	return mat3x3(vec3(1.0 - 2.0 * (_S72 + _S73), 2.0 * (_S74 - _S75), 2.0 * (_S76 + _S77)),
	              vec3(2.0 * (_S74 + _S75), 1.0 - 2.0 * (_S78 + _S73), 2.0 * (_S79 - _S80)),
	              vec3(2.0 * (_S76 - _S77), 2.0 * (_S79 + _S80), 1.0 - 2.0 * (_S78 + _S72)));
}
mat3x3 s_primal_ctx_mul_1(mat3x3 _S81, mat3x3 _S82) { return (((_S82) * (_S81))); }
struct DiffPair_SH_0 {
	SH primal_0;
	SH differential_0;
};
void s_bwd_prop_max_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S83, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S84,
                      vec3 _S85) {
	_d_max_vector_0(_S83, _S84, _S85);
	return;
}
void s_bwd_prop_sh2color_0(inout DiffPair_SH_0 dpsh_0, inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpdir_0, vec3 _s_dOut_0) {
	float x_3 = dpdir_0.primal_0.x;
	float y_3 = dpdir_0.primal_0.y;
	float z_3 = dpdir_0.primal_0.z;
	float xx_1 = x_3 * x_3;
	float yy_1 = y_3 * y_3;
	float zz_1 = z_3 * z_3;
	float xy_1 = x_3 * y_3;
	float _S86 = -0.48860251903533936 * y_3;
	vec3 _S87 = vec3(_S86);
	float _S88 = 0.48860251903533936 * z_3;
	vec3 _S89 = vec3(_S88);
	float _S90 = 0.48860251903533936 * x_3;
	vec3 _S91 = vec3(_S90);
	float _S92 = 1.09254848957061768 * xy_1;
	vec3 _S93 = vec3(_S92);
	float _S94 = -1.09254848957061768 * (y_3 * z_3);
	vec3 _S95 = vec3(_S94);
	float _S96 = 2.0 * zz_1;
	float _S97 = 0.31539157032966614 * (_S96 - xx_1 - yy_1);
	vec3 _S98 = vec3(_S97);
	float _S99 = -1.09254848957061768 * (x_3 * z_3);
	vec3 _S100 = vec3(_S99);
	float _S101 = xx_1 - yy_1;
	float _S102 = 0.54627424478530884 * _S101;
	vec3 _S103 = vec3(_S102);
	float _S104 = -0.59004360437393188 * y_3;
	float _S105 = 3.0 * xx_1;
	float _S106 = _S105 - yy_1;
	float _S107 = _S104 * _S106;
	vec3 _S108 = vec3(_S107);
	float _S109 = 2.89061141014099121 * xy_1;
	float _S110 = _S109 * z_3;
	vec3 _S111 = vec3(_S110);
	float _S112 = -0.4570457935333252 * y_3;
	float _S113 = 4.0 * zz_1 - xx_1 - yy_1;
	float _S114 = _S112 * _S113;
	vec3 _S115 = vec3(_S114);
	float _S116 = 0.37317633628845215 * z_3;
	float _S117 = 3.0 * yy_1;
	float _S118 = _S96 - _S105 - _S117;
	float _S119 = _S116 * _S118;
	vec3 _S120 = vec3(_S119);
	float _S121 = -0.4570457935333252 * x_3;
	float _S122 = _S121 * _S113;
	vec3 _S123 = vec3(_S122);
	float _S124 = 1.44530570507049561 * z_3;
	float _S125 = _S124 * _S101;
	vec3 _S126 = vec3(_S125);
	float _S127 = -0.59004360437393188 * x_3;
	float _S128 = xx_1 - _S117;
	float _S129 = _S127 * _S128;
	vec3 _S130 = vec3(_S129);
	const vec3 _S131 = vec3(0.0);
	vec3 _S132 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S133;
	_S133.primal_0 =
	    0.282094806432724 * dpsh_0.primal_0.data[0] + 0.5 +
	    (_S86 * dpsh_0.primal_0.data[1] + _S88 * dpsh_0.primal_0.data[2] - _S90 * dpsh_0.primal_0.data[3]) +
	    (_S92 * dpsh_0.primal_0.data[4] + _S94 * dpsh_0.primal_0.data[5] + _S97 * dpsh_0.primal_0.data[6] +
	     _S99 * dpsh_0.primal_0.data[7] + _S102 * dpsh_0.primal_0.data[8]) +
	    (_S107 * dpsh_0.primal_0.data[9] + _S110 * dpsh_0.primal_0.data[10] + _S114 * dpsh_0.primal_0.data[11] +
	     _S119 * dpsh_0.primal_0.data[12] + _S122 * dpsh_0.primal_0.data[13] + _S125 * dpsh_0.primal_0.data[14] +
	     _S129 * dpsh_0.primal_0.data[15]);
	_S133.differential_0 = _S132;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S134;
	_S134.primal_0 = _S131;
	_S134.differential_0 = _S132;
	s_bwd_prop_max_0(_S133, _S134, _s_dOut_0);
	vec3 _S135 = _S130 * _S133.differential_0;
	vec3 _S136 = dpsh_0.primal_0.data[15] * _S133.differential_0;
	float _S137 = _S136[0] + _S136[1] + _S136[2];
	float _S138 = _S127 * _S137;
	vec3 _S139 = _S126 * _S133.differential_0;
	vec3 _S140 = dpsh_0.primal_0.data[14] * _S133.differential_0;
	float _S141 = _S140[0] + _S140[1] + _S140[2];
	vec3 _S142 = _S123 * _S133.differential_0;
	vec3 _S143 = dpsh_0.primal_0.data[13] * _S133.differential_0;
	float _S144 = _S143[0] + _S143[1] + _S143[2];
	vec3 _S145 = _S120 * _S133.differential_0;
	vec3 _S146 = dpsh_0.primal_0.data[12] * _S133.differential_0;
	float _S147 = _S146[0] + _S146[1] + _S146[2];
	float _S148 = _S116 * _S147;
	float _S149 = -_S148;
	vec3 _S150 = _S115 * _S133.differential_0;
	vec3 _S151 = dpsh_0.primal_0.data[11] * _S133.differential_0;
	float _S152 = _S151[0] + _S151[1] + _S151[2];
	float _S153 = _S121 * _S144 + _S112 * _S152;
	float _S154 = -_S153;
	vec3 _S155 = _S111 * _S133.differential_0;
	vec3 _S156 = dpsh_0.primal_0.data[10] * _S133.differential_0;
	float _S157 = _S156[0] + _S156[1] + _S156[2];
	vec3 _S158 = _S108 * _S133.differential_0;
	vec3 _S159 = dpsh_0.primal_0.data[9] * _S133.differential_0;
	float _S160 = _S159[0] + _S159[1] + _S159[2];
	float _S161 = _S104 * _S160;
	vec3 _S162 = _S103 * _S133.differential_0;
	vec3 _S163 = dpsh_0.primal_0.data[8] * _S133.differential_0;
	float _S164 = _S124 * _S141 + 0.54627424478530884 * (_S163[0] + _S163[1] + _S163[2]);
	vec3 _S165 = _S100 * _S133.differential_0;
	vec3 _S166 = dpsh_0.primal_0.data[7] * _S133.differential_0;
	float s_diff_xz_T_0 = -1.09254848957061768 * (_S166[0] + _S166[1] + _S166[2]);
	vec3 _S167 = _S98 * _S133.differential_0;
	vec3 _S168 = dpsh_0.primal_0.data[6] * _S133.differential_0;
	float _S169 = 0.31539157032966614 * (_S168[0] + _S168[1] + _S168[2]);
	float _S170 = -_S169;
	vec3 _S171 = _S95 * _S133.differential_0;
	vec3 _S172 = dpsh_0.primal_0.data[5] * _S133.differential_0;
	float s_diff_yz_T_0 = -1.09254848957061768 * (_S172[0] + _S172[1] + _S172[2]);
	vec3 _S173 = _S93 * _S133.differential_0;
	vec3 _S174 = dpsh_0.primal_0.data[4] * _S133.differential_0;
	vec3 _S175 = -_S133.differential_0;
	vec3 _S176 = _S91 * _S175;
	vec3 _S177 = dpsh_0.primal_0.data[3] * _S175;
	vec3 _S178 = _S89 * _S133.differential_0;
	vec3 _S179 = dpsh_0.primal_0.data[2] * _S133.differential_0;
	vec3 _S180 = _S87 * _S133.differential_0;
	vec3 _S181 = dpsh_0.primal_0.data[1] * _S133.differential_0;
	float _S182 = 2.89061141014099121 * (z_3 * _S157) + 1.09254848957061768 * (_S174[0] + _S174[1] + _S174[2]);
	float _S183 = z_3 * (4.0 * _S153 + 2.0 * (_S148 + _S169));
	float _S184 = y_3 * (3.0 * (-_S138 + _S149) + _S154 + -_S161 + -_S164 + _S170);
	float _S185 = x_3 * (_S138 + _S154 + 3.0 * (_S149 + _S161) + _S164 + _S170);
	float _S186 = 1.44530570507049561 * (_S101 * _S141) + 0.37317633628845215 * (_S118 * _S147) + _S109 * _S157 +
	              0.48860251903533936 * (_S179[0] + _S179[1] + _S179[2]) + x_3 * s_diff_xz_T_0 + y_3 * s_diff_yz_T_0 +
	              _S183 + _S183;
	float _S187 = -0.4570457935333252 * (_S113 * _S152) + -0.59004360437393188 * (_S106 * _S160) +
	              -0.48860251903533936 * (_S181[0] + _S181[1] + _S181[2]) + z_3 * s_diff_yz_T_0 + x_3 * _S182 + _S184 +
	              _S184;
	float _S188 = -0.59004360437393188 * (_S128 * _S137) + -0.4570457935333252 * (_S113 * _S144) +
	              0.48860251903533936 * (_S177[0] + _S177[1] + _S177[2]) + z_3 * s_diff_xz_T_0 + y_3 * _S182 + _S185 +
	              _S185;
	vec3 _S189 = vec3(0.282094806432724) * _S133.differential_0;
	vec3 _S190[16];
	_S190[0] = _S132;
	_S190[1] = _S132;
	_S190[2] = _S132;
	_S190[3] = _S132;
	_S190[4] = _S132;
	_S190[5] = _S132;
	_S190[6] = _S132;
	_S190[7] = _S132;
	_S190[8] = _S132;
	_S190[9] = _S132;
	_S190[10] = _S132;
	_S190[11] = _S132;
	_S190[12] = _S132;
	_S190[13] = _S132;
	_S190[14] = _S132;
	_S190[15] = _S132;
	_S190[15] = _S135;
	_S190[14] = _S139;
	_S190[13] = _S142;
	_S190[12] = _S145;
	_S190[11] = _S150;
	_S190[10] = _S155;
	_S190[9] = _S158;
	_S190[8] = _S162;
	_S190[7] = _S165;
	_S190[6] = _S167;
	_S190[5] = _S171;
	_S190[4] = _S173;
	_S190[3] = _S176;
	_S190[2] = _S178;
	_S190[1] = _S180;
	_S190[0] = _S189;
	vec3 _S191 = vec3(_S188, _S187, _S186);
	dpdir_0.primal_0 = dpdir_0.primal_0;
	dpdir_0.differential_0 = _S191;
	SH _S192 = SH_x24_syn_dzero_0();
	_S192.data = _S190;
	dpsh_0.primal_0 = dpsh_0.primal_0;
	dpsh_0.differential_0 = _S192;
	return;
}
void s_bwd_prop_sqrt_0(inout DiffPair_float_0 _S193, float _S194) {
	_d_sqrt_0(_S193, _S194);
	return;
}
void s_bwd_prop_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_5, float _s_dOut_1) {
	float _S195 = dpx_5.primal_0[0];
	float _S196 = dpx_5.primal_0[1];
	float _S197 = dpx_5.primal_0[2];
	DiffPair_float_0 _S198;
	_S198.primal_0 = _S195 * _S195 + _S196 * _S196 + _S197 * _S197;
	_S198.differential_0 = 0.0;
	s_bwd_prop_sqrt_0(_S198, _s_dOut_1);
	float _S199 = dpx_5.primal_0[2] * _S198.differential_0;
	float _S200 = _S199 + _S199;
	float _S201 = dpx_5.primal_0[1] * _S198.differential_0;
	float _S202 = _S201 + _S201;
	float _S203 = dpx_5.primal_0[0] * _S198.differential_0;
	float _S204 = _S203 + _S203;
	vec3 _S205 = vec3(0.0);
	_S205[2] = _S200;
	_S205[1] = _S202;
	_S205[0] = _S204;
	dpx_5.primal_0 = dpx_5.primal_0;
	dpx_5.differential_0 = _S205;
	return;
}
void s_bwd_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S206, float _S207) {
	s_bwd_prop_length_impl_0(_S206, _S207);
	return;
}
void s_bwd_prop_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_6, vec3 _s_dOut_2) {
	float _S208 = length(dpx_6.primal_0);
	vec3 _S209 = dpx_6.primal_0 * _s_dOut_2;
	vec3 _S210 = vec3(1.0 / _S208) * _s_dOut_2;
	float _S211 = -((_S209[0] + _S209[1] + _S209[2]) / (_S208 * _S208));
	vec3 _S212 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S213;
	_S213.primal_0 = dpx_6.primal_0;
	_S213.differential_0 = _S212;
	s_bwd_length_impl_0(_S213, _S211);
	vec3 _S214 = _S210 + _S213.differential_0;
	dpx_6.primal_0 = dpx_6.primal_0;
	dpx_6.differential_0 = _S214;
	return;
}
void s_bwd_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S215, vec3 _S216) {
	s_bwd_prop_normalize_impl_0(_S215, _S216);
	return;
}
void s_bwd_prop_cov2conic_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpcov_0, vec3 _s_dOut_3) {
	float _S217 = dpcov_0.primal_0.x;
	float _S218 = dpcov_0.primal_0.z;
	float _S219 = dpcov_0.primal_0.y;
	float det_0 = _S217 * _S218 - _S219 * _S219;
	vec3 _S220 = _s_dOut_3 / vec3(det_0 * det_0);
	vec3 _S221 = vec3(_S218, -_S219, _S217) * -_S220;
	vec3 _S222 = vec3(det_0) * _S220;
	float _S223 = _S221[0] + _S221[1] + _S221[2];
	float _S224 = _S219 * -_S223;
	vec3 _S225 = vec3(_S222[2] + _S218 * _S223, -_S222[1] + _S224 + _S224, _S222[0] + _S217 * _S223);
	dpcov_0.primal_0 = dpcov_0.primal_0;
	dpcov_0.differential_0 = _S225;
	return;
}
void s_bwd_prop_mul_0(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S226,
                      inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S227, mat3x3 _S228) {
	mul_0(_S226, _S227, _S228);
	return;
}
struct DiffPair_vectorx3Cfloatx2C4x3E_0 {
	vec4 primal_0;
	vec4 differential_0;
};
void s_bwd_prop_quat2matrix_0(inout DiffPair_vectorx3Cfloatx2C4x3E_0 dpquat_1, mat3x3 _s_dOut_4) {
	float x_4 = dpquat_1.primal_0.x;
	float y_4 = dpquat_1.primal_0.y;
	float z_4 = dpquat_1.primal_0.z;
	float w_2 = dpquat_1.primal_0.w;
	float _S229 = 2.0 * -_s_dOut_4[2][2];
	float _S230 = 2.0 * _s_dOut_4[2][1];
	float _S231 = 2.0 * _s_dOut_4[2][0];
	float _S232 = 2.0 * _s_dOut_4[1][2];
	float _S233 = _S230 + -_S232;
	float _S234 = _S230 + _S232;
	float _S235 = 2.0 * -_s_dOut_4[1][1];
	float _S236 = x_4 * (_S229 + _S235);
	float _S237 = 2.0 * _s_dOut_4[1][0];
	float _S238 = 2.0 * _s_dOut_4[0][2];
	float _S239 = -_S231 + _S238;
	float _S240 = _S231 + _S238;
	float _S241 = 2.0 * _s_dOut_4[0][1];
	float _S242 = _S237 + -_S241;
	float _S243 = _S237 + _S241;
	float _S244 = 2.0 * -_s_dOut_4[0][0];
	float _S245 = z_4 * (_S235 + _S244);
	float _S246 = y_4 * (_S229 + _S244);
	vec4 _S247 = vec4(w_2 * _S233 + _S236 + _S236 + z_4 * _S240 + y_4 * _S243,
	                  z_4 * _S234 + w_2 * _S239 + x_4 * _S243 + _S246 + _S246,
	                  y_4 * _S234 + x_4 * _S240 + w_2 * _S242 + _S245 + _S245, x_4 * _S233 + y_4 * _S239 + z_4 * _S242);
	dpquat_1.primal_0 = dpquat_1.primal_0;
	dpquat_1.differential_0 = _S247;
	return;
}
void s_bwd_prop_scale2matrix_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpscale_1, mat3x3 _s_dOut_5) {
	vec3 _S248 = vec3(_s_dOut_5[0][0], _s_dOut_5[1][1], _s_dOut_5[2][2]);
	dpscale_1.primal_0 = dpscale_1.primal_0;
	dpscale_1.differential_0 = _S248;
	return;
}
void s_bwd_prop_clamp_0(inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S249, inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S250,
                        inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S251, vec2 _S252) {
	_d_clamp_vector_0(_S249, _S250, _S251, _S252);
	return;
}
void s_bwd_prop_mul_1(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S253, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S254,
                      vec3 _S255) {
	_d_mul_0(_S253, _S254, _S255);
	return;
}
void s_bwd_prop_splat2splatView_0(inout DiffPair_Splat_0 dpsplat_0, Camera camera_1, SplatView _s_dOut_6) {
	vec3 camMean_1 = dpsplat_0.primal_0.geom.mean - camera_1.pos;
	vec3 _S256 = s_primal_ctx_mul_0(camera_1.viewMat, camMean_1);
	float _S257 = _S256.z;
	float invViewMeanZ_1 = 1.0 / _S257;
	vec2 _S258 = vec2(invViewMeanZ_1);
	float _S259 = _S257 * _S257;
	vec2 _S260 = _S256.xy * camera_1.focal;
	vec2 projMean_1 = _S260 * invViewMeanZ_1;
	vec2 camHalfRes_1 = vec2(camera_1.resolution) * 0.5;
	vec2 _S261 = -1.29999995231628418 * camHalfRes_1;
	vec2 _S262 = 1.29999995231628418 * camHalfRes_1;
	vec2 _S263 = s_primal_ctx_clamp_0(projMean_1, _S261, _S262);
	mat3x3 _S264 = s_primal_ctx_scale2matrix_0(dpsplat_0.primal_0.geom.scale);
	mat3x3 _S265 = s_primal_ctx_quat2matrix_0(dpsplat_0.primal_0.geom.quat);
	float _S266 = camera_1.focal.x;
	float _S267 = -_S263.x;
	float _S268 = camera_1.focal.y;
	float _S269 = -_S263.y;
	mat3x3 J_0 = mat3x3(vec3(_S266 * invViewMeanZ_1, 0.0, _S267 * invViewMeanZ_1),
	                    vec3(0.0, _S268 * invViewMeanZ_1, _S269 * invViewMeanZ_1), vec3(0.0, 0.0, 0.0));
	mat3x3 _S270 = s_primal_ctx_mul_1(J_0, camera_1.viewMat);
	mat3x3 _S271 = s_primal_ctx_mul_1(_S265, _S264);
	mat3x3 _S272 = s_primal_ctx_mul_1(_S270, _S271);
	mat3x3 _S273 = transpose(_S272);
	mat3x3 _S274 = s_primal_ctx_mul_1(_S272, _S273);
	mat2x2 _S275 = mat2x2(_S274[0].xy, _S274[1].xy);
	vec3 cov2D_1 = vec3(_S275[0][0], _S275[0][1], _S275[1][1]) + vec3(0.30000001192092896, 0.0, 0.30000001192092896);
	vec3 _S276 = normalize(camMean_1);
	SH _S277 = SH_x24_syn_dzero_0();
	DiffPair_SH_0 _S278;
	_S278.primal_0 = dpsplat_0.primal_0.sh;
	_S278.differential_0 = _S277;
	vec3 _S279 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S280;
	_S280.primal_0 = _S276;
	_S280.differential_0 = _S279;
	s_bwd_prop_sh2color_0(_S278, _S280, _s_dOut_6.color);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S281;
	_S281.primal_0 = camMean_1;
	_S281.differential_0 = _S279;
	s_bwd_normalize_impl_0(_S281, _S280.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S282;
	_S282.primal_0 = cov2D_1;
	_S282.differential_0 = _S279;
	s_bwd_prop_cov2conic_0(_S282, _s_dOut_6.geom.conic);
	vec2 _S283 = vec2(0.0);
	vec2 _S284 = _S283;
	_S284[1] = _S282.differential_0[2];
	vec2 _S285 = _S283;
	_S285[1] = _S282.differential_0[1];
	_S285[0] = _S282.differential_0[0];
	mat2x2 _S286 = mat2x2(0.0, 0.0, 0.0, 0.0);
	_S286[1] = _S284;
	_S286[0] = _S285;
	vec3 _S287 = vec3(_S286[1][0], _S286[1][1], 0.0);
	vec3 _S288 = vec3(_S286[0][0], _S286[0][1], 0.0);
	mat3x3 _S289 = mat3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	mat3x3 _S290 = _S289;
	_S290[1] = _S287;
	_S290[0] = _S288;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S291;
	_S291.primal_0 = _S272;
	_S291.differential_0 = _S289;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S292;
	_S292.primal_0 = _S273;
	_S292.differential_0 = _S289;
	s_bwd_prop_mul_0(_S291, _S292, _S290);
	mat3x3 _S293 = _S291.differential_0 + transpose(_S292.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S294;
	_S294.primal_0 = _S270;
	_S294.differential_0 = _S289;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S295;
	_S295.primal_0 = _S271;
	_S295.differential_0 = _S289;
	s_bwd_prop_mul_0(_S294, _S295, _S293);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S296;
	_S296.primal_0 = _S265;
	_S296.differential_0 = _S289;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S297;
	_S297.primal_0 = _S264;
	_S297.differential_0 = _S289;
	s_bwd_prop_mul_0(_S296, _S297, _S295.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S298;
	_S298.primal_0 = J_0;
	_S298.differential_0 = _S289;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S299;
	_S299.primal_0 = camera_1.viewMat;
	_S299.differential_0 = _S289;
	s_bwd_prop_mul_0(_S298, _S299, _S294.differential_0);
	float _S300 = _S269 * _S298.differential_0[1][2];
	float _S301 = -(invViewMeanZ_1 * _S298.differential_0[1][2]);
	float _S302 = _S268 * _S298.differential_0[1][1];
	float _S303 = _S267 * _S298.differential_0[0][2];
	float _S304 = -(invViewMeanZ_1 * _S298.differential_0[0][2]);
	float _S305 = _S266 * _S298.differential_0[0][0];
	vec4 _S306 = vec4(0.0);
	DiffPair_vectorx3Cfloatx2C4x3E_0 _S307;
	_S307.primal_0 = dpsplat_0.primal_0.geom.quat;
	_S307.differential_0 = _S306;
	s_bwd_prop_quat2matrix_0(_S307, _S296.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S308;
	_S308.primal_0 = dpsplat_0.primal_0.geom.scale;
	_S308.differential_0 = _S279;
	s_bwd_prop_scale2matrix_0(_S308, _S297.differential_0);
	vec2 _S309 = vec2(_S304, _S301);
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S310;
	_S310.primal_0 = projMean_1;
	_S310.differential_0 = _S283;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S311;
	_S311.primal_0 = _S261;
	_S311.differential_0 = _S283;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S312;
	_S312.primal_0 = _S262;
	_S312.differential_0 = _S283;
	s_bwd_prop_clamp_0(_S310, _S311, _S312, _S309);
	vec2 _S313 = _s_dOut_6.geom.mean2D + _S310.differential_0;
	vec2 _S314 = _S260 * _S313;
	vec2 _S315 = camera_1.focal * (_S258 * _S313);
	vec3 _S316 = vec3(_S315[0], _S315[1], -((_S300 + _S302 + _S303 + _S305 + _S314[0] + _S314[1]) / _S259));
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S317;
	_S317.primal_0 = camera_1.viewMat;
	_S317.differential_0 = _S289;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S318;
	_S318.primal_0 = camMean_1;
	_S318.differential_0 = _S279;
	s_bwd_prop_mul_1(_S317, _S318, _S316);
	vec3 _S319 = _S281.differential_0 + _S318.differential_0;
	SplatGeom _S320 = SplatGeom_x24_syn_dzero_0();
	_S320.opacity = _s_dOut_6.geom.opacity;
	_S320.quat = _S307.differential_0;
	_S320.scale = _S308.differential_0;
	_S320.mean = _S319;
	Splat _S321 = Splat_x24_syn_dzero_0();
	_S321.sh = _S278.differential_0;
	_S321.geom = _S320;
	dpsplat_0.primal_0 = dpsplat_0.primal_0;
	dpsplat_0.differential_0 = _S321;
	return;
}
void s_bwd_splat2splatView_0(inout DiffPair_Splat_0 _S322, Camera _S323, SplatView _S324) {
	s_bwd_prop_splat2splatView_0(_S322, _S323, _S324);
	return;
}
Splat bwd_splat2splatView(Splat splat_1, Camera camera_2, SplatView dL_dsplatView_0) {
	Splat _S325 = Splat_x24_syn_dzero_0();
	DiffPair_Splat_0 dp_0;
	dp_0.primal_0 = splat_1;
	dp_0.differential_0 = _S325;
	s_bwd_splat2splatView_0(dp_0, camera_2, dL_dsplatView_0);
	return dp_0.differential_0;
}
void _d_exp_0(inout DiffPair_float_0 dpx_7, float dOut_7) {
	float _S326 = exp(dpx_7.primal_0) * dOut_7;
	dpx_7.primal_0 = dpx_7.primal_0;
	dpx_7.differential_0 = _S326;
	return;
}
float splatViewGeom2alpha(SplatViewGeom splatViewGeom_0, vec2 fragCoord_0, Camera camera_3) {
	vec2 x2D_0 = vec2(fragCoord_0.x, float(camera_3.resolution.y) - fragCoord_0.y) - splatViewGeom_0.mean2D;
	float _S327 = x2D_0.x;
	float _S328 = x2D_0.y;
	return splatViewGeom_0.opacity *
	       exp(-0.5 * (splatViewGeom_0.conic.x * _S327 * _S327 + splatViewGeom_0.conic.z * _S328 * _S328) -
	           splatViewGeom_0.conic.y * _S327 * _S328);
}
SplatViewGeom SplatViewGeom_x24_syn_dzero_0() {
	SplatViewGeom result_3;
	result_3.conic = vec3(0.0);
	result_3.mean2D = vec2(0.0);
	result_3.opacity = 0.0;
	return result_3;
}
struct DiffPair_SplatViewGeom_0 {
	SplatViewGeom primal_0;
	SplatViewGeom differential_0;
};
float s_primal_ctx_exp_0(float _S329) { return exp(_S329); }
void s_bwd_prop_exp_0(inout DiffPair_float_0 _S330, float _S331) {
	_d_exp_0(_S330, _S331);
	return;
}
void s_bwd_prop_splatViewGeom2alpha_0(inout DiffPair_SplatViewGeom_0 dpsplatViewGeom_0, vec2 fragCoord_1,
                                      Camera camera_4, float _s_dOut_7) {
	vec2 x2D_1 = vec2(fragCoord_1.x, float(camera_4.resolution.y) - fragCoord_1.y) - dpsplatViewGeom_0.primal_0.mean2D;
	float _S332 = dpsplatViewGeom_0.primal_0.conic.x;
	float _S333 = x2D_1.x;
	float _S334 = _S332 * _S333;
	float _S335 = dpsplatViewGeom_0.primal_0.conic.z;
	float _S336 = x2D_1.y;
	float _S337 = _S335 * _S336;
	float _S338 = dpsplatViewGeom_0.primal_0.conic.y;
	float _S339 = _S338 * _S333;
	float power_0 = -0.5 * (_S334 * _S333 + _S337 * _S336) - _S339 * _S336;
	float _S340 = dpsplatViewGeom_0.primal_0.opacity * _s_dOut_7;
	float _S341 = s_primal_ctx_exp_0(power_0) * _s_dOut_7;
	DiffPair_float_0 _S342;
	_S342.primal_0 = power_0;
	_S342.differential_0 = 0.0;
	s_bwd_prop_exp_0(_S342, _S340);
	float _S343 = -_S342.differential_0;
	float _S344 = _S336 * _S343;
	float _S345 = -0.5 * _S342.differential_0;
	float _S346 = _S336 * _S345;
	float _S347 = _S333 * _S345;
	vec3 _S348 = vec3(_S333 * _S347, _S333 * _S344, _S336 * _S346);
	vec2 _S349 = -vec2(_S338 * _S344 + _S334 * _S345 + _S332 * _S347, _S339 * _S343 + _S337 * _S345 + _S335 * _S346);
	SplatViewGeom _S350 = SplatViewGeom_x24_syn_dzero_0();
	_S350.opacity = _S341;
	_S350.conic = _S348;
	_S350.mean2D = _S349;
	dpsplatViewGeom_0.primal_0 = dpsplatViewGeom_0.primal_0;
	dpsplatViewGeom_0.differential_0 = _S350;
	return;
}
void s_bwd_splatViewGeom2alpha_0(inout DiffPair_SplatViewGeom_0 _S351, vec2 _S352, Camera _S353, float _S354) {
	s_bwd_prop_splatViewGeom2alpha_0(_S351, _S352, _S353, _S354);
	return;
}
SplatViewGeom bwd_splatViewGeom2alpha(SplatViewGeom splatViewGeom_1, vec2 fragCoord_2, Camera camera_5,
                                      float dL_dalpha_0) {
	SplatViewGeom _S355 = SplatViewGeom_x24_syn_dzero_0();
	DiffPair_SplatViewGeom_0 dp_1;
	dp_1.primal_0 = splatViewGeom_1;
	dp_1.differential_0 = _S355;
	s_bwd_splatViewGeom2alpha_0(dp_1, fragCoord_2, camera_5, dL_dalpha_0);
	return dp_1.differential_0;
}
bool behindFrustum(Splat splat_2, Camera camera_6) {
	return ((((splat_2.geom.mean - camera_6.pos) * (camera_6.viewMat))).z) < 0.20000000298023224;
}
bool inFrustum(SplatViewGeom splatViewGeom_2, SplatQuad splatQuad_0, Camera camera_7) { return true; }
#endif
