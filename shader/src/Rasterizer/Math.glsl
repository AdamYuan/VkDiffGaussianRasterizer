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
	float y_0 = quat[2];
	float _S45 = y_0 * y_0;
	float _S46 = quat[3] * quat[3];
	float _S47 = quat[1] * quat[2];
	float _S48 = quat[0] * quat[3];
	float _S49 = quat[1] * quat[3];
	float _S50 = quat[0] * quat[2];
	float _S51 = quat[1] * quat[1];
	float _S52 = quat[2] * quat[3];
	float _S53 = quat[0] * quat[1];
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
	float x_0 = dir_0.x;
	float y_1 = dir_0.y;
	float z_0 = dir_0.z;
	float xx_0 = x_0 * x_0;
	float yy_0 = y_1 * y_1;
	float zz_0 = z_0 * z_0;
	float xy_0 = x_0 * y_1;
	float _S58 = 2.0 * zz_0;
	float _S59 = xx_0 - yy_0;
	float _S60 = 3.0 * xx_0;
	float _S61 = 4.0 * zz_0 - xx_0 - yy_0;
	float _S62 = 3.0 * yy_0;
	return max(0.282094806432724 * sh.data[0] + 0.5 +
	               (-0.48860251903533936 * y_1 * sh.data[1] + 0.48860251903533936 * z_0 * sh.data[2] -
	                0.48860251903533936 * x_0 * sh.data[3]) +
	               (1.09254848957061768 * xy_0 * sh.data[4] + -1.09254848957061768 * (y_1 * z_0) * sh.data[5] +
	                0.31539157032966614 * (_S58 - xx_0 - yy_0) * sh.data[6] +
	                -1.09254848957061768 * (x_0 * z_0) * sh.data[7] + 0.54627424478530884 * _S59 * sh.data[8]) +
	               (-0.59004360437393188 * y_1 * (_S60 - yy_0) * sh.data[9] +
	                2.89061141014099121 * xy_0 * z_0 * sh.data[10] + -0.4570457935333252 * y_1 * _S61 * sh.data[11] +
	                0.37317633628845215 * z_0 * (_S58 - _S60 - _S62) * sh.data[12] +
	                -0.4570457935333252 * x_0 * _S61 * sh.data[13] + 1.44530570507049561 * z_0 * _S59 * sh.data[14] +
	                -0.59004360437393188 * x_0 * (xx_0 - _S62) * sh.data[15]),
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
	splatView_0.geom.mean2D = projMean_0;
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
	float _S72 = dpquat_0[2];
	float _S73 = _S72 * _S72;
	float _S74 = dpquat_0[3] * dpquat_0[3];
	float _S75 = dpquat_0[1] * dpquat_0[2];
	float _S76 = dpquat_0[0] * dpquat_0[3];
	float _S77 = dpquat_0[1] * dpquat_0[3];
	float _S78 = dpquat_0[0] * dpquat_0[2];
	float _S79 = dpquat_0[1] * dpquat_0[1];
	float _S80 = dpquat_0[2] * dpquat_0[3];
	float _S81 = dpquat_0[0] * dpquat_0[1];
	return mat3x3(vec3(1.0 - 2.0 * (_S73 + _S74), 2.0 * (_S75 - _S76), 2.0 * (_S77 + _S78)),
	              vec3(2.0 * (_S75 + _S76), 1.0 - 2.0 * (_S79 + _S74), 2.0 * (_S80 - _S81)),
	              vec3(2.0 * (_S77 - _S78), 2.0 * (_S80 + _S81), 1.0 - 2.0 * (_S79 + _S73)));
}
mat3x3 s_primal_ctx_mul_1(mat3x3 _S82, mat3x3 _S83) { return (((_S83) * (_S82))); }
struct DiffPair_SH_0 {
	SH primal_0;
	SH differential_0;
};
void s_bwd_prop_max_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S84, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S85,
                      vec3 _S86) {
	_d_max_vector_0(_S84, _S85, _S86);
	return;
}
void s_bwd_prop_sh2color_0(inout DiffPair_SH_0 dpsh_0, inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpdir_0, vec3 _s_dOut_0) {
	float x_1 = dpdir_0.primal_0.x;
	float y_2 = dpdir_0.primal_0.y;
	float z_1 = dpdir_0.primal_0.z;
	float xx_1 = x_1 * x_1;
	float yy_1 = y_2 * y_2;
	float zz_1 = z_1 * z_1;
	float xy_1 = x_1 * y_2;
	float _S87 = -0.48860251903533936 * y_2;
	vec3 _S88 = vec3(_S87);
	float _S89 = 0.48860251903533936 * z_1;
	vec3 _S90 = vec3(_S89);
	float _S91 = 0.48860251903533936 * x_1;
	vec3 _S92 = vec3(_S91);
	float _S93 = 1.09254848957061768 * xy_1;
	vec3 _S94 = vec3(_S93);
	float _S95 = -1.09254848957061768 * (y_2 * z_1);
	vec3 _S96 = vec3(_S95);
	float _S97 = 2.0 * zz_1;
	float _S98 = 0.31539157032966614 * (_S97 - xx_1 - yy_1);
	vec3 _S99 = vec3(_S98);
	float _S100 = -1.09254848957061768 * (x_1 * z_1);
	vec3 _S101 = vec3(_S100);
	float _S102 = xx_1 - yy_1;
	float _S103 = 0.54627424478530884 * _S102;
	vec3 _S104 = vec3(_S103);
	float _S105 = -0.59004360437393188 * y_2;
	float _S106 = 3.0 * xx_1;
	float _S107 = _S106 - yy_1;
	float _S108 = _S105 * _S107;
	vec3 _S109 = vec3(_S108);
	float _S110 = 2.89061141014099121 * xy_1;
	float _S111 = _S110 * z_1;
	vec3 _S112 = vec3(_S111);
	float _S113 = -0.4570457935333252 * y_2;
	float _S114 = 4.0 * zz_1 - xx_1 - yy_1;
	float _S115 = _S113 * _S114;
	vec3 _S116 = vec3(_S115);
	float _S117 = 0.37317633628845215 * z_1;
	float _S118 = 3.0 * yy_1;
	float _S119 = _S97 - _S106 - _S118;
	float _S120 = _S117 * _S119;
	vec3 _S121 = vec3(_S120);
	float _S122 = -0.4570457935333252 * x_1;
	float _S123 = _S122 * _S114;
	vec3 _S124 = vec3(_S123);
	float _S125 = 1.44530570507049561 * z_1;
	float _S126 = _S125 * _S102;
	vec3 _S127 = vec3(_S126);
	float _S128 = -0.59004360437393188 * x_1;
	float _S129 = xx_1 - _S118;
	float _S130 = _S128 * _S129;
	vec3 _S131 = vec3(_S130);
	const vec3 _S132 = vec3(0.0);
	vec3 _S133 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S134;
	_S134.primal_0 =
	    0.282094806432724 * dpsh_0.primal_0.data[0] + 0.5 +
	    (_S87 * dpsh_0.primal_0.data[1] + _S89 * dpsh_0.primal_0.data[2] - _S91 * dpsh_0.primal_0.data[3]) +
	    (_S93 * dpsh_0.primal_0.data[4] + _S95 * dpsh_0.primal_0.data[5] + _S98 * dpsh_0.primal_0.data[6] +
	     _S100 * dpsh_0.primal_0.data[7] + _S103 * dpsh_0.primal_0.data[8]) +
	    (_S108 * dpsh_0.primal_0.data[9] + _S111 * dpsh_0.primal_0.data[10] + _S115 * dpsh_0.primal_0.data[11] +
	     _S120 * dpsh_0.primal_0.data[12] + _S123 * dpsh_0.primal_0.data[13] + _S126 * dpsh_0.primal_0.data[14] +
	     _S130 * dpsh_0.primal_0.data[15]);
	_S134.differential_0 = _S133;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S135;
	_S135.primal_0 = _S132;
	_S135.differential_0 = _S133;
	s_bwd_prop_max_0(_S134, _S135, _s_dOut_0);
	vec3 _S136 = _S131 * _S134.differential_0;
	vec3 _S137 = dpsh_0.primal_0.data[15] * _S134.differential_0;
	float _S138 = _S137[0] + _S137[1] + _S137[2];
	float _S139 = _S128 * _S138;
	vec3 _S140 = _S127 * _S134.differential_0;
	vec3 _S141 = dpsh_0.primal_0.data[14] * _S134.differential_0;
	float _S142 = _S141[0] + _S141[1] + _S141[2];
	vec3 _S143 = _S124 * _S134.differential_0;
	vec3 _S144 = dpsh_0.primal_0.data[13] * _S134.differential_0;
	float _S145 = _S144[0] + _S144[1] + _S144[2];
	vec3 _S146 = _S121 * _S134.differential_0;
	vec3 _S147 = dpsh_0.primal_0.data[12] * _S134.differential_0;
	float _S148 = _S147[0] + _S147[1] + _S147[2];
	float _S149 = _S117 * _S148;
	float _S150 = -_S149;
	vec3 _S151 = _S116 * _S134.differential_0;
	vec3 _S152 = dpsh_0.primal_0.data[11] * _S134.differential_0;
	float _S153 = _S152[0] + _S152[1] + _S152[2];
	float _S154 = _S122 * _S145 + _S113 * _S153;
	float _S155 = -_S154;
	vec3 _S156 = _S112 * _S134.differential_0;
	vec3 _S157 = dpsh_0.primal_0.data[10] * _S134.differential_0;
	float _S158 = _S157[0] + _S157[1] + _S157[2];
	vec3 _S159 = _S109 * _S134.differential_0;
	vec3 _S160 = dpsh_0.primal_0.data[9] * _S134.differential_0;
	float _S161 = _S160[0] + _S160[1] + _S160[2];
	float _S162 = _S105 * _S161;
	vec3 _S163 = _S104 * _S134.differential_0;
	vec3 _S164 = dpsh_0.primal_0.data[8] * _S134.differential_0;
	float _S165 = _S125 * _S142 + 0.54627424478530884 * (_S164[0] + _S164[1] + _S164[2]);
	vec3 _S166 = _S101 * _S134.differential_0;
	vec3 _S167 = dpsh_0.primal_0.data[7] * _S134.differential_0;
	float s_diff_xz_T_0 = -1.09254848957061768 * (_S167[0] + _S167[1] + _S167[2]);
	vec3 _S168 = _S99 * _S134.differential_0;
	vec3 _S169 = dpsh_0.primal_0.data[6] * _S134.differential_0;
	float _S170 = 0.31539157032966614 * (_S169[0] + _S169[1] + _S169[2]);
	float _S171 = -_S170;
	vec3 _S172 = _S96 * _S134.differential_0;
	vec3 _S173 = dpsh_0.primal_0.data[5] * _S134.differential_0;
	float s_diff_yz_T_0 = -1.09254848957061768 * (_S173[0] + _S173[1] + _S173[2]);
	vec3 _S174 = _S94 * _S134.differential_0;
	vec3 _S175 = dpsh_0.primal_0.data[4] * _S134.differential_0;
	vec3 _S176 = -_S134.differential_0;
	vec3 _S177 = _S92 * _S176;
	vec3 _S178 = dpsh_0.primal_0.data[3] * _S176;
	vec3 _S179 = _S90 * _S134.differential_0;
	vec3 _S180 = dpsh_0.primal_0.data[2] * _S134.differential_0;
	vec3 _S181 = _S88 * _S134.differential_0;
	vec3 _S182 = dpsh_0.primal_0.data[1] * _S134.differential_0;
	float _S183 = 2.89061141014099121 * (z_1 * _S158) + 1.09254848957061768 * (_S175[0] + _S175[1] + _S175[2]);
	float _S184 = z_1 * (4.0 * _S154 + 2.0 * (_S149 + _S170));
	float _S185 = y_2 * (3.0 * (-_S139 + _S150) + _S155 + -_S162 + -_S165 + _S171);
	float _S186 = x_1 * (_S139 + _S155 + 3.0 * (_S150 + _S162) + _S165 + _S171);
	float _S187 = 1.44530570507049561 * (_S102 * _S142) + 0.37317633628845215 * (_S119 * _S148) + _S110 * _S158 +
	              0.48860251903533936 * (_S180[0] + _S180[1] + _S180[2]) + x_1 * s_diff_xz_T_0 + y_2 * s_diff_yz_T_0 +
	              _S184 + _S184;
	float _S188 = -0.4570457935333252 * (_S114 * _S153) + -0.59004360437393188 * (_S107 * _S161) +
	              -0.48860251903533936 * (_S182[0] + _S182[1] + _S182[2]) + z_1 * s_diff_yz_T_0 + x_1 * _S183 + _S185 +
	              _S185;
	float _S189 = -0.59004360437393188 * (_S129 * _S138) + -0.4570457935333252 * (_S114 * _S145) +
	              0.48860251903533936 * (_S178[0] + _S178[1] + _S178[2]) + z_1 * s_diff_xz_T_0 + y_2 * _S183 + _S186 +
	              _S186;
	vec3 _S190 = vec3(0.282094806432724) * _S134.differential_0;
	vec3 _S191[16];
	_S191[0] = _S133;
	_S191[1] = _S133;
	_S191[2] = _S133;
	_S191[3] = _S133;
	_S191[4] = _S133;
	_S191[5] = _S133;
	_S191[6] = _S133;
	_S191[7] = _S133;
	_S191[8] = _S133;
	_S191[9] = _S133;
	_S191[10] = _S133;
	_S191[11] = _S133;
	_S191[12] = _S133;
	_S191[13] = _S133;
	_S191[14] = _S133;
	_S191[15] = _S133;
	_S191[15] = _S136;
	_S191[14] = _S140;
	_S191[13] = _S143;
	_S191[12] = _S146;
	_S191[11] = _S151;
	_S191[10] = _S156;
	_S191[9] = _S159;
	_S191[8] = _S163;
	_S191[7] = _S166;
	_S191[6] = _S168;
	_S191[5] = _S172;
	_S191[4] = _S174;
	_S191[3] = _S177;
	_S191[2] = _S179;
	_S191[1] = _S181;
	_S191[0] = _S190;
	vec3 _S192 = vec3(_S189, _S188, _S187);
	dpdir_0.primal_0 = dpdir_0.primal_0;
	dpdir_0.differential_0 = _S192;
	SH _S193 = SH_x24_syn_dzero_0();
	_S193.data = _S191;
	dpsh_0.primal_0 = dpsh_0.primal_0;
	dpsh_0.differential_0 = _S193;
	return;
}
void s_bwd_prop_sqrt_0(inout DiffPair_float_0 _S194, float _S195) {
	_d_sqrt_0(_S194, _S195);
	return;
}
void s_bwd_prop_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_5, float _s_dOut_1) {
	float _S196 = dpx_5.primal_0[0];
	float _S197 = dpx_5.primal_0[1];
	float _S198 = dpx_5.primal_0[2];
	DiffPair_float_0 _S199;
	_S199.primal_0 = _S196 * _S196 + _S197 * _S197 + _S198 * _S198;
	_S199.differential_0 = 0.0;
	s_bwd_prop_sqrt_0(_S199, _s_dOut_1);
	float _S200 = dpx_5.primal_0[2] * _S199.differential_0;
	float _S201 = _S200 + _S200;
	float _S202 = dpx_5.primal_0[1] * _S199.differential_0;
	float _S203 = _S202 + _S202;
	float _S204 = dpx_5.primal_0[0] * _S199.differential_0;
	float _S205 = _S204 + _S204;
	vec3 _S206 = vec3(0.0);
	_S206[2] = _S201;
	_S206[1] = _S203;
	_S206[0] = _S205;
	dpx_5.primal_0 = dpx_5.primal_0;
	dpx_5.differential_0 = _S206;
	return;
}
void s_bwd_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S207, float _S208) {
	s_bwd_prop_length_impl_0(_S207, _S208);
	return;
}
void s_bwd_prop_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_6, vec3 _s_dOut_2) {
	float _S209 = length(dpx_6.primal_0);
	vec3 _S210 = dpx_6.primal_0 * _s_dOut_2;
	vec3 _S211 = vec3(1.0 / _S209) * _s_dOut_2;
	float _S212 = -((_S210[0] + _S210[1] + _S210[2]) / (_S209 * _S209));
	vec3 _S213 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S214;
	_S214.primal_0 = dpx_6.primal_0;
	_S214.differential_0 = _S213;
	s_bwd_length_impl_0(_S214, _S212);
	vec3 _S215 = _S211 + _S214.differential_0;
	dpx_6.primal_0 = dpx_6.primal_0;
	dpx_6.differential_0 = _S215;
	return;
}
void s_bwd_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S216, vec3 _S217) {
	s_bwd_prop_normalize_impl_0(_S216, _S217);
	return;
}
void s_bwd_prop_cov2conic_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpcov_0, vec3 _s_dOut_3) {
	float _S218 = dpcov_0.primal_0.x;
	float _S219 = dpcov_0.primal_0.z;
	float _S220 = dpcov_0.primal_0.y;
	float det_0 = _S218 * _S219 - _S220 * _S220;
	vec3 _S221 = _s_dOut_3 / vec3(det_0 * det_0);
	vec3 _S222 = vec3(_S219, -_S220, _S218) * -_S221;
	vec3 _S223 = vec3(det_0) * _S221;
	float _S224 = _S222[0] + _S222[1] + _S222[2];
	float _S225 = _S220 * -_S224;
	vec3 _S226 = vec3(_S223[2] + _S219 * _S224, -_S223[1] + _S225 + _S225, _S223[0] + _S218 * _S224);
	dpcov_0.primal_0 = dpcov_0.primal_0;
	dpcov_0.differential_0 = _S226;
	return;
}
void s_bwd_prop_mul_0(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S227,
                      inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S228, mat3x3 _S229) {
	mul_0(_S227, _S228, _S229);
	return;
}
struct DiffPair_vectorx3Cfloatx2C4x3E_0 {
	vec4 primal_0;
	vec4 differential_0;
};
void s_bwd_prop_quat2matrix_0(inout DiffPair_vectorx3Cfloatx2C4x3E_0 dpquat_1, mat3x3 _s_dOut_4) {
	float _S230 = 2.0 * -_s_dOut_4[2][2];
	float _S231 = 2.0 * _s_dOut_4[2][1];
	float _S232 = 2.0 * _s_dOut_4[2][0];
	float _S233 = 2.0 * _s_dOut_4[1][2];
	float _S234 = _S231 + -_S233;
	float _S235 = _S231 + _S233;
	float _S236 = 2.0 * -_s_dOut_4[1][1];
	float _S237 = dpquat_1.primal_0[1] * (_S230 + _S236);
	float _S238 = 2.0 * _s_dOut_4[1][0];
	float _S239 = 2.0 * _s_dOut_4[0][2];
	float _S240 = -_S232 + _S239;
	float _S241 = _S232 + _S239;
	float _S242 = 2.0 * _s_dOut_4[0][1];
	float _S243 = _S238 + -_S242;
	float _S244 = _S238 + _S242;
	float _S245 = 2.0 * -_s_dOut_4[0][0];
	float _S246 = dpquat_1.primal_0[3] * (_S236 + _S245);
	float _S247 = dpquat_1.primal_0[2] * (_S230 + _S245);
	float _S248 = dpquat_1.primal_0[1] * _S234 + dpquat_1.primal_0[2] * _S240 + dpquat_1.primal_0[3] * _S243;
	float _S249 =
	    dpquat_1.primal_0[2] * _S235 + dpquat_1.primal_0[1] * _S241 + dpquat_1.primal_0[0] * _S243 + _S246 + _S246;
	float _S250 =
	    dpquat_1.primal_0[3] * _S235 + dpquat_1.primal_0[0] * _S240 + dpquat_1.primal_0[1] * _S244 + _S247 + _S247;
	float _S251 =
	    dpquat_1.primal_0[0] * _S234 + _S237 + _S237 + dpquat_1.primal_0[3] * _S241 + dpquat_1.primal_0[2] * _S244;
	vec4 _S252 = vec4(0.0);
	_S252[0] = _S248;
	_S252[3] = _S249;
	_S252[2] = _S250;
	_S252[1] = _S251;
	dpquat_1.primal_0 = dpquat_1.primal_0;
	dpquat_1.differential_0 = _S252;
	return;
}
void s_bwd_prop_scale2matrix_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpscale_1, mat3x3 _s_dOut_5) {
	vec3 _S253 = vec3(_s_dOut_5[0][0], _s_dOut_5[1][1], _s_dOut_5[2][2]);
	dpscale_1.primal_0 = dpscale_1.primal_0;
	dpscale_1.differential_0 = _S253;
	return;
}
void s_bwd_prop_clamp_0(inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S254, inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S255,
                        inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S256, vec2 _S257) {
	_d_clamp_vector_0(_S254, _S255, _S256, _S257);
	return;
}
void s_bwd_prop_mul_1(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S258, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S259,
                      vec3 _S260) {
	_d_mul_0(_S258, _S259, _S260);
	return;
}
void s_bwd_prop_splat2splatView_0(inout DiffPair_Splat_0 dpsplat_0, Camera camera_1, SplatView _s_dOut_6) {
	vec3 camMean_1 = dpsplat_0.primal_0.geom.mean - camera_1.pos;
	vec3 _S261 = s_primal_ctx_mul_0(camera_1.viewMat, camMean_1);
	float _S262 = _S261.z;
	float invViewMeanZ_1 = 1.0 / _S262;
	vec2 _S263 = vec2(invViewMeanZ_1);
	float _S264 = _S262 * _S262;
	vec2 _S265 = _S261.xy * camera_1.focal;
	vec2 projMean_1 = _S265 * invViewMeanZ_1;
	vec2 camHalfRes_1 = vec2(camera_1.resolution) * 0.5;
	vec2 _S266 = -1.29999995231628418 * camHalfRes_1;
	vec2 _S267 = 1.29999995231628418 * camHalfRes_1;
	vec2 _S268 = s_primal_ctx_clamp_0(projMean_1, _S266, _S267);
	mat3x3 _S269 = s_primal_ctx_scale2matrix_0(dpsplat_0.primal_0.geom.scale);
	mat3x3 _S270 = s_primal_ctx_quat2matrix_0(dpsplat_0.primal_0.geom.quat);
	float _S271 = camera_1.focal.x;
	float _S272 = -_S268.x;
	float _S273 = camera_1.focal.y;
	float _S274 = -_S268.y;
	mat3x3 J_0 = mat3x3(vec3(_S271 * invViewMeanZ_1, 0.0, _S272 * invViewMeanZ_1),
	                    vec3(0.0, _S273 * invViewMeanZ_1, _S274 * invViewMeanZ_1), vec3(0.0, 0.0, 0.0));
	mat3x3 _S275 = s_primal_ctx_mul_1(J_0, camera_1.viewMat);
	mat3x3 _S276 = s_primal_ctx_mul_1(_S270, _S269);
	mat3x3 _S277 = s_primal_ctx_mul_1(_S275, _S276);
	mat3x3 _S278 = transpose(_S277);
	mat3x3 _S279 = s_primal_ctx_mul_1(_S277, _S278);
	mat2x2 _S280 = mat2x2(_S279[0].xy, _S279[1].xy);
	vec3 cov2D_1 = vec3(_S280[0][0], _S280[0][1], _S280[1][1]) + vec3(0.30000001192092896, 0.0, 0.30000001192092896);
	vec3 _S281 = normalize(camMean_1);
	SH _S282 = SH_x24_syn_dzero_0();
	DiffPair_SH_0 _S283;
	_S283.primal_0 = dpsplat_0.primal_0.sh;
	_S283.differential_0 = _S282;
	vec3 _S284 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S285;
	_S285.primal_0 = _S281;
	_S285.differential_0 = _S284;
	s_bwd_prop_sh2color_0(_S283, _S285, _s_dOut_6.color);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S286;
	_S286.primal_0 = camMean_1;
	_S286.differential_0 = _S284;
	s_bwd_normalize_impl_0(_S286, _S285.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S287;
	_S287.primal_0 = cov2D_1;
	_S287.differential_0 = _S284;
	s_bwd_prop_cov2conic_0(_S287, _s_dOut_6.geom.conic);
	vec2 _S288 = vec2(0.0);
	vec2 _S289 = _S288;
	_S289[1] = _S287.differential_0[2];
	vec2 _S290 = _S288;
	_S290[1] = _S287.differential_0[1];
	_S290[0] = _S287.differential_0[0];
	mat2x2 _S291 = mat2x2(0.0, 0.0, 0.0, 0.0);
	_S291[1] = _S289;
	_S291[0] = _S290;
	vec3 _S292 = vec3(_S291[1][0], _S291[1][1], 0.0);
	vec3 _S293 = vec3(_S291[0][0], _S291[0][1], 0.0);
	mat3x3 _S294 = mat3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	mat3x3 _S295 = _S294;
	_S295[1] = _S292;
	_S295[0] = _S293;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S296;
	_S296.primal_0 = _S277;
	_S296.differential_0 = _S294;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S297;
	_S297.primal_0 = _S278;
	_S297.differential_0 = _S294;
	s_bwd_prop_mul_0(_S296, _S297, _S295);
	mat3x3 _S298 = _S296.differential_0 + transpose(_S297.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S299;
	_S299.primal_0 = _S275;
	_S299.differential_0 = _S294;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S300;
	_S300.primal_0 = _S276;
	_S300.differential_0 = _S294;
	s_bwd_prop_mul_0(_S299, _S300, _S298);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S301;
	_S301.primal_0 = _S270;
	_S301.differential_0 = _S294;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S302;
	_S302.primal_0 = _S269;
	_S302.differential_0 = _S294;
	s_bwd_prop_mul_0(_S301, _S302, _S300.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S303;
	_S303.primal_0 = J_0;
	_S303.differential_0 = _S294;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S304;
	_S304.primal_0 = camera_1.viewMat;
	_S304.differential_0 = _S294;
	s_bwd_prop_mul_0(_S303, _S304, _S299.differential_0);
	float _S305 = _S274 * _S303.differential_0[1][2];
	float _S306 = -(invViewMeanZ_1 * _S303.differential_0[1][2]);
	float _S307 = _S273 * _S303.differential_0[1][1];
	float _S308 = _S272 * _S303.differential_0[0][2];
	float _S309 = -(invViewMeanZ_1 * _S303.differential_0[0][2]);
	float _S310 = _S271 * _S303.differential_0[0][0];
	vec4 _S311 = vec4(0.0);
	DiffPair_vectorx3Cfloatx2C4x3E_0 _S312;
	_S312.primal_0 = dpsplat_0.primal_0.geom.quat;
	_S312.differential_0 = _S311;
	s_bwd_prop_quat2matrix_0(_S312, _S301.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S313;
	_S313.primal_0 = dpsplat_0.primal_0.geom.scale;
	_S313.differential_0 = _S284;
	s_bwd_prop_scale2matrix_0(_S313, _S302.differential_0);
	vec2 _S314 = vec2(_S309, _S306);
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S315;
	_S315.primal_0 = projMean_1;
	_S315.differential_0 = _S288;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S316;
	_S316.primal_0 = _S266;
	_S316.differential_0 = _S288;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S317;
	_S317.primal_0 = _S267;
	_S317.differential_0 = _S288;
	s_bwd_prop_clamp_0(_S315, _S316, _S317, _S314);
	vec2 _S318 = _s_dOut_6.geom.mean2D + _S315.differential_0;
	vec2 _S319 = _S265 * _S318;
	vec2 _S320 = camera_1.focal * (_S263 * _S318);
	vec3 _S321 = vec3(_S320[0], _S320[1], -((_S305 + _S307 + _S308 + _S310 + _S319[0] + _S319[1]) / _S264));
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S322;
	_S322.primal_0 = camera_1.viewMat;
	_S322.differential_0 = _S294;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S323;
	_S323.primal_0 = camMean_1;
	_S323.differential_0 = _S284;
	s_bwd_prop_mul_1(_S322, _S323, _S321);
	vec3 _S324 = _S286.differential_0 + _S323.differential_0;
	SplatGeom _S325 = SplatGeom_x24_syn_dzero_0();
	_S325.opacity = _s_dOut_6.geom.opacity;
	_S325.quat = _S312.differential_0;
	_S325.scale = _S313.differential_0;
	_S325.mean = _S324;
	Splat _S326 = Splat_x24_syn_dzero_0();
	_S326.sh = _S283.differential_0;
	_S326.geom = _S325;
	dpsplat_0.primal_0 = dpsplat_0.primal_0;
	dpsplat_0.differential_0 = _S326;
	return;
}
void s_bwd_splat2splatView_0(inout DiffPair_Splat_0 _S327, Camera _S328, SplatView _S329) {
	s_bwd_prop_splat2splatView_0(_S327, _S328, _S329);
	return;
}
Splat bwd_splat2splatView(Splat splat_1, Camera camera_2, SplatView dL_dsplatView_0) {
	Splat _S330 = Splat_x24_syn_dzero_0();
	DiffPair_Splat_0 dp_0;
	dp_0.primal_0 = splat_1;
	dp_0.differential_0 = _S330;
	s_bwd_splat2splatView_0(dp_0, camera_2, dL_dsplatView_0);
	return dp_0.differential_0;
}
float splatViewGeom2alpha(SplatViewGeom splatViewGeom_0, vec2 fragCoord_0, Camera camera_3) {
	vec2 d_0 = fragCoord_0 - vec2(camera_3.resolution) * 0.5 - splatViewGeom_0.mean2D;
	float _S331 = d_0.x;
	float _S332 = d_0.y;
	return splatViewGeom_0.opacity *
	       exp(-0.5 * (splatViewGeom_0.conic.x * _S331 * _S331 + splatViewGeom_0.conic.z * _S332 * _S332) -
	           splatViewGeom_0.conic.y * _S331 * _S332);
}
bool behindFrustum(Splat splat_2, Camera camera_4, out float o_viewMeanZ_0) {
	float _S333 = (((splat_2.geom.mean - camera_4.pos) * (camera_4.viewMat))).z;
	o_viewMeanZ_0 = _S333;
	return _S333 < 0.20000000298023224;
}
float opacity2quadBound(float opacity) { return sqrt(2.0 * (5.54126358032226562 + log(opacity))); }
bool inFrustum(SplatViewGeom splatViewGeom_1, SplatQuad splatQuad_0, Camera camera_5) {
	vec2 camHalfRes_2 = vec2(camera_5.resolution) * 0.5;
	vec2 quadExtent_0 = opacity2quadBound(splatViewGeom_1.opacity) * (abs(splatQuad_0.axis1) + abs(splatQuad_0.axis2));
	bool _S334;
	if ((any(bvec2((greaterThan(splatViewGeom_1.mean2D - quadExtent_0, camHalfRes_2)))))) {
		_S334 = true;
	} else {
		_S334 = (any(bvec2((lessThan(splatViewGeom_1.mean2D + quadExtent_0, -camHalfRes_2)))));
	}
	if (_S334) {
		return false;
	}
	return true;
}
vec2 pos2D2clip(vec2 pos, Camera camera_6) { return pos * (2.0 / vec2(camera_6.resolution)); }
vec2 axis2D2clip(vec2 axis_0, Camera camera_7) { return axis_0 * (2.0 / vec2(camera_7.resolution)); }
float quadPos2alpha(vec2 quadPos_0, float opacity, out float o_G_0) {
	float G_0 = exp(-0.5 * dot(quadPos_0, quadPos_0));
	o_G_0 = G_0;
	return opacity * G_0;
}
float quadPos2alpha(vec2 quadPos_1, float opacity) {
	float G_1;
	return quadPos2alpha(quadPos_1, opacity, G_1);
}
SplatViewGeom bwd_splatViewGeom2alpha(SplatViewGeom splatViewGeom_2, vec2 fragCoord_1, Camera camera_8, float G_2,
                                      float dL_dalpha_0) {
	float dL_dG_0 = splatViewGeom_2.opacity * dL_dalpha_0;
	vec2 d_1 = splatViewGeom_2.mean2D - (fragCoord_1 - vec2(camera_8.resolution) * 0.5);
	float _S335 = d_1.x;
	float gdx_0 = G_2 * _S335;
	float _S336 = d_1.y;
	float gdy_0 = G_2 * _S336;
	float _S337 = -gdx_0;
	float _S338 = splatViewGeom_2.conic.y;
	float dG_ddely_0 = -gdy_0 * splatViewGeom_2.conic.z - gdx_0 * _S338;
	SplatViewGeom dL_dGeom_0;
	dL_dGeom_0.mean2D[0] = dL_dG_0 * (_S337 * splatViewGeom_2.conic.x - gdy_0 * _S338);
	dL_dGeom_0.mean2D[1] = dL_dG_0 * dG_ddely_0;
	dL_dGeom_0.conic[0] = -0.5 * gdx_0 * _S335 * dL_dG_0;
	dL_dGeom_0.conic[1] = _S337 * _S336 * dL_dG_0;
	dL_dGeom_0.conic[2] = -0.5 * gdy_0 * _S336 * dL_dG_0;
	dL_dGeom_0.opacity = G_2 * dL_dalpha_0;
	return dL_dGeom_0;
}
#endif
