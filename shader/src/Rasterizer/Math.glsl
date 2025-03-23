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
	float _S32 = left_0.primal_0[0][0] * dOut_0[0];
	left_d_result_0[0][0] = right_0.primal_0[0] * dOut_0[0];
	float sum_0 = _S32 + left_0.primal_0[1][0] * dOut_0[1];
	left_d_result_0[1][0] = right_0.primal_0[0] * dOut_0[1];
	float sum_1 = sum_0 + left_0.primal_0[2][0] * dOut_0[2];
	left_d_result_0[2][0] = right_0.primal_0[0] * dOut_0[2];
	right_d_result_0[0] = sum_1;
	float _S33 = left_0.primal_0[0][1] * dOut_0[0];
	left_d_result_0[0][1] = right_0.primal_0[1] * dOut_0[0];
	float sum_2 = _S33 + left_0.primal_0[1][1] * dOut_0[1];
	left_d_result_0[1][1] = right_0.primal_0[1] * dOut_0[1];
	float sum_3 = sum_2 + left_0.primal_0[2][1] * dOut_0[2];
	left_d_result_0[2][1] = right_0.primal_0[1] * dOut_0[2];
	right_d_result_0[1] = sum_3;
	float _S34 = left_0.primal_0[0][2] * dOut_0[0];
	left_d_result_0[0][2] = right_0.primal_0[2] * dOut_0[0];
	float sum_4 = _S34 + left_0.primal_0[1][2] * dOut_0[1];
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
	DiffPair_float_0 _S35 = dpx_0;
	float _S36;
	if ((dpx_0.primal_0) > (dpy_0.primal_0)) {
		_S36 = dOut_1;
	} else {
		_S36 = 0.0;
	}
	dpx_0.primal_0 = _S35.primal_0;
	dpx_0.differential_0 = _S36;
	DiffPair_float_0 _S37 = dpy_0;
	if ((dpy_0.primal_0) > (_S35.primal_0)) {
		_S36 = dOut_1;
	} else {
		_S36 = 0.0;
	}
	dpy_0.primal_0 = _S37.primal_0;
	dpy_0.differential_0 = _S36;
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
	DiffPair_float_0 _S38 = dpx_2;
	bool _S39;
	if ((dpx_2.primal_0) > (dpMin_0.primal_0)) {
		_S39 = (dpx_2.primal_0) < (dpMax_0.primal_0);
	} else {
		_S39 = false;
	}
	float _S40;
	if (_S39) {
		_S40 = dOut_3;
	} else {
		_S40 = 0.0;
	}
	dpx_2.primal_0 = _S38.primal_0;
	dpx_2.differential_0 = _S40;
	DiffPair_float_0 _S41 = dpMin_0;
	if ((_S38.primal_0) <= (dpMin_0.primal_0)) {
		_S40 = dOut_3;
	} else {
		_S40 = 0.0;
	}
	dpMin_0.primal_0 = _S41.primal_0;
	dpMin_0.differential_0 = _S40;
	DiffPair_float_0 _S42 = dpMax_0;
	if ((dpx_2.primal_0) >= (dpMax_0.primal_0)) {
		_S40 = dOut_3;
	} else {
		_S40 = 0.0;
	}
	dpMax_0.primal_0 = _S42.primal_0;
	dpMax_0.differential_0 = _S40;
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
	float _S43 = y_0 * y_0;
	float _S44 = z_0 * z_0;
	float _S45 = x_0 * y_0;
	float _S46 = w_0 * z_0;
	float _S47 = x_0 * z_0;
	float _S48 = w_0 * y_0;
	float _S49 = x_0 * x_0;
	float _S50 = y_0 * z_0;
	float _S51 = w_0 * x_0;
	return mat3x3(vec3(1.0 - 2.0 * (_S43 + _S44), 2.0 * (_S45 - _S46), 2.0 * (_S47 + _S48)),
	              vec3(2.0 * (_S45 + _S46), 1.0 - 2.0 * (_S49 + _S44), 2.0 * (_S50 - _S51)),
	              vec3(2.0 * (_S47 - _S48), 2.0 * (_S50 + _S51), 1.0 - 2.0 * (_S49 + _S43)));
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
	float _S52 = cov_0.x;
	float _S53 = cov_0.z;
	float _S54 = cov_0.y;
	return vec3(_S53, -_S54, _S52) / (_S52 * _S53 - _S54 * _S54);
}
void _d_sqrt_0(inout DiffPair_float_0 dpx_4, float dOut_6) {
	float _S55 = 0.5 / sqrt(max(1.00000001168609742e-07, dpx_4.primal_0)) * dOut_6;
	dpx_4.primal_0 = dpx_4.primal_0;
	dpx_4.differential_0 = _S55;
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
	float _S56 = 2.0 * zz_0;
	float _S57 = xx_0 - yy_0;
	float _S58 = 3.0 * xx_0;
	float _S59 = 4.0 * zz_0 - xx_0 - yy_0;
	float _S60 = 3.0 * yy_0;
	return max(0.282094806432724 * sh.data[0] + 0.5 +
	               (-0.48860251903533936 * y_1 * sh.data[1] + 0.48860251903533936 * z_1 * sh.data[2] -
	                0.48860251903533936 * x_1 * sh.data[3]) +
	               (1.09254848957061768 * xy_0 * sh.data[4] + -1.09254848957061768 * (y_1 * z_1) * sh.data[5] +
	                0.31539157032966614 * (_S56 - xx_0 - yy_0) * sh.data[6] +
	                -1.09254848957061768 * (x_1 * z_1) * sh.data[7] + 0.54627424478530884 * _S57 * sh.data[8]) +
	               (-0.59004360437393188 * y_1 * (_S58 - yy_0) * sh.data[9] +
	                2.89061141014099121 * xy_0 * z_1 * sh.data[10] + -0.4570457935333252 * y_1 * _S59 * sh.data[11] +
	                0.37317633628845215 * z_1 * (_S56 - _S58 - _S60) * sh.data[12] +
	                -0.4570457935333252 * x_1 * _S59 * sh.data[13] + 1.44530570507049561 * z_1 * _S57 * sh.data[14] +
	                -0.59004360437393188 * x_1 * (xx_0 - _S60) * sh.data[15]),
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
SplatView splat2splatView(Splat splat_0, vec3 camPos_0, mat3x3 camViewMat_0, vec2 camFocal_0, vec2 camResolution_0,
                          out SplatQuad o_splatQuad_0) {
	vec3 camMean_0 = splat_0.geom.mean - camPos_0;
	vec3 viewMean_0 = (((camMean_0) * (camViewMat_0)));
	float invViewMeanZ_0 = 1.0 / viewMean_0.z;
	vec2 projMean_0 = viewMean_0.xy * camFocal_0 * invViewMeanZ_0;
	vec2 camHalfRes_0 = camResolution_0 * 0.5;
	vec2 clampedProjMean_0 = clamp(projMean_0, -1.29999995231628418 * camHalfRes_0, 1.29999995231628418 * camHalfRes_0);
	mat3x3 JWRS_0 =
	    ((((((scale2matrix_0(splat_0.geom.scale)) * (quat2matrix_0(splat_0.geom.quat))))) *
	      ((((camViewMat_0) * (mat3x3(vec3(camFocal_0.x * invViewMeanZ_0, 0.0, -clampedProjMean_0.x * invViewMeanZ_0),
	                                  vec3(0.0, camFocal_0.y * invViewMeanZ_0, -clampedProjMean_0.y * invViewMeanZ_0),
	                                  vec3(0.0, 0.0, 0.0))))))));
	mat3x3 _S61 = (((transpose(JWRS_0)) * (JWRS_0)));
	mat2x2 _S62 = mat2x2(_S61[0].xy, _S61[1].xy);
	vec3 cov2D_0 = vec3(_S62[0][0], _S62[0][1], _S62[1][1]) + vec3(0.30000001192092896, 0.0, 0.30000001192092896);
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
	const vec3 _S63 = vec3(0.0);
	result_0.scale = _S63;
	result_0.mean = _S63;
	result_0.opacity = 0.0;
	return result_0;
}
SH SH_x24_syn_dzero_0() {
	SH result_1;
	const vec3 _S64 = vec3(0.0);
	result_1.data[0] = _S64;
	result_1.data[1] = _S64;
	result_1.data[2] = _S64;
	result_1.data[3] = _S64;
	result_1.data[4] = _S64;
	result_1.data[5] = _S64;
	result_1.data[6] = _S64;
	result_1.data[7] = _S64;
	result_1.data[8] = _S64;
	result_1.data[9] = _S64;
	result_1.data[10] = _S64;
	result_1.data[11] = _S64;
	result_1.data[12] = _S64;
	result_1.data[13] = _S64;
	result_1.data[14] = _S64;
	result_1.data[15] = _S64;
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
vec3 s_primal_ctx_mul_0(mat3x3 _S65, vec3 _S66) { return (((_S66) * (_S65))); }
vec2 s_primal_ctx_clamp_0(vec2 _S67, vec2 _S68, vec2 _S69) { return clamp(_S67, _S68, _S69); }
mat3x3 s_primal_ctx_scale2matrix_0(vec3 dpscale_0) {
	return mat3x3(dpscale_0.x, 0.0, 0.0, 0.0, dpscale_0.y, 0.0, 0.0, 0.0, dpscale_0.z);
}
mat3x3 s_primal_ctx_quat2matrix_0(vec4 dpquat_0) {
	float x_2 = dpquat_0.x;
	float y_2 = dpquat_0.y;
	float z_2 = dpquat_0.z;
	float w_1 = dpquat_0.w;
	float _S70 = y_2 * y_2;
	float _S71 = z_2 * z_2;
	float _S72 = x_2 * y_2;
	float _S73 = w_1 * z_2;
	float _S74 = x_2 * z_2;
	float _S75 = w_1 * y_2;
	float _S76 = x_2 * x_2;
	float _S77 = y_2 * z_2;
	float _S78 = w_1 * x_2;
	return mat3x3(vec3(1.0 - 2.0 * (_S70 + _S71), 2.0 * (_S72 - _S73), 2.0 * (_S74 + _S75)),
	              vec3(2.0 * (_S72 + _S73), 1.0 - 2.0 * (_S76 + _S71), 2.0 * (_S77 - _S78)),
	              vec3(2.0 * (_S74 - _S75), 2.0 * (_S77 + _S78), 1.0 - 2.0 * (_S76 + _S70)));
}
mat3x3 s_primal_ctx_mul_1(mat3x3 _S79, mat3x3 _S80) { return (((_S80) * (_S79))); }
struct DiffPair_SH_0 {
	SH primal_0;
	SH differential_0;
};
void s_bwd_prop_max_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S81, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S82,
                      vec3 _S83) {
	_d_max_vector_0(_S81, _S82, _S83);
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
	float _S84 = -0.48860251903533936 * y_3;
	vec3 _S85 = vec3(_S84);
	float _S86 = 0.48860251903533936 * z_3;
	vec3 _S87 = vec3(_S86);
	float _S88 = 0.48860251903533936 * x_3;
	vec3 _S89 = vec3(_S88);
	float _S90 = 1.09254848957061768 * xy_1;
	vec3 _S91 = vec3(_S90);
	float _S92 = -1.09254848957061768 * (y_3 * z_3);
	vec3 _S93 = vec3(_S92);
	float _S94 = 2.0 * zz_1;
	float _S95 = 0.31539157032966614 * (_S94 - xx_1 - yy_1);
	vec3 _S96 = vec3(_S95);
	float _S97 = -1.09254848957061768 * (x_3 * z_3);
	vec3 _S98 = vec3(_S97);
	float _S99 = xx_1 - yy_1;
	float _S100 = 0.54627424478530884 * _S99;
	vec3 _S101 = vec3(_S100);
	float _S102 = -0.59004360437393188 * y_3;
	float _S103 = 3.0 * xx_1;
	float _S104 = _S103 - yy_1;
	float _S105 = _S102 * _S104;
	vec3 _S106 = vec3(_S105);
	float _S107 = 2.89061141014099121 * xy_1;
	float _S108 = _S107 * z_3;
	vec3 _S109 = vec3(_S108);
	float _S110 = -0.4570457935333252 * y_3;
	float _S111 = 4.0 * zz_1 - xx_1 - yy_1;
	float _S112 = _S110 * _S111;
	vec3 _S113 = vec3(_S112);
	float _S114 = 0.37317633628845215 * z_3;
	float _S115 = 3.0 * yy_1;
	float _S116 = _S94 - _S103 - _S115;
	float _S117 = _S114 * _S116;
	vec3 _S118 = vec3(_S117);
	float _S119 = -0.4570457935333252 * x_3;
	float _S120 = _S119 * _S111;
	vec3 _S121 = vec3(_S120);
	float _S122 = 1.44530570507049561 * z_3;
	float _S123 = _S122 * _S99;
	vec3 _S124 = vec3(_S123);
	float _S125 = -0.59004360437393188 * x_3;
	float _S126 = xx_1 - _S115;
	float _S127 = _S125 * _S126;
	vec3 _S128 = vec3(_S127);
	const vec3 _S129 = vec3(0.0);
	vec3 _S130 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S131;
	_S131.primal_0 =
	    0.282094806432724 * dpsh_0.primal_0.data[0] + 0.5 +
	    (_S84 * dpsh_0.primal_0.data[1] + _S86 * dpsh_0.primal_0.data[2] - _S88 * dpsh_0.primal_0.data[3]) +
	    (_S90 * dpsh_0.primal_0.data[4] + _S92 * dpsh_0.primal_0.data[5] + _S95 * dpsh_0.primal_0.data[6] +
	     _S97 * dpsh_0.primal_0.data[7] + _S100 * dpsh_0.primal_0.data[8]) +
	    (_S105 * dpsh_0.primal_0.data[9] + _S108 * dpsh_0.primal_0.data[10] + _S112 * dpsh_0.primal_0.data[11] +
	     _S117 * dpsh_0.primal_0.data[12] + _S120 * dpsh_0.primal_0.data[13] + _S123 * dpsh_0.primal_0.data[14] +
	     _S127 * dpsh_0.primal_0.data[15]);
	_S131.differential_0 = _S130;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S132;
	_S132.primal_0 = _S129;
	_S132.differential_0 = _S130;
	s_bwd_prop_max_0(_S131, _S132, _s_dOut_0);
	vec3 _S133 = _S128 * _S131.differential_0;
	vec3 _S134 = dpsh_0.primal_0.data[15] * _S131.differential_0;
	float _S135 = _S134[0] + _S134[1] + _S134[2];
	float _S136 = _S125 * _S135;
	vec3 _S137 = _S124 * _S131.differential_0;
	vec3 _S138 = dpsh_0.primal_0.data[14] * _S131.differential_0;
	float _S139 = _S138[0] + _S138[1] + _S138[2];
	vec3 _S140 = _S121 * _S131.differential_0;
	vec3 _S141 = dpsh_0.primal_0.data[13] * _S131.differential_0;
	float _S142 = _S141[0] + _S141[1] + _S141[2];
	vec3 _S143 = _S118 * _S131.differential_0;
	vec3 _S144 = dpsh_0.primal_0.data[12] * _S131.differential_0;
	float _S145 = _S144[0] + _S144[1] + _S144[2];
	float _S146 = _S114 * _S145;
	float _S147 = -_S146;
	vec3 _S148 = _S113 * _S131.differential_0;
	vec3 _S149 = dpsh_0.primal_0.data[11] * _S131.differential_0;
	float _S150 = _S149[0] + _S149[1] + _S149[2];
	float _S151 = _S119 * _S142 + _S110 * _S150;
	float _S152 = -_S151;
	vec3 _S153 = _S109 * _S131.differential_0;
	vec3 _S154 = dpsh_0.primal_0.data[10] * _S131.differential_0;
	float _S155 = _S154[0] + _S154[1] + _S154[2];
	vec3 _S156 = _S106 * _S131.differential_0;
	vec3 _S157 = dpsh_0.primal_0.data[9] * _S131.differential_0;
	float _S158 = _S157[0] + _S157[1] + _S157[2];
	float _S159 = _S102 * _S158;
	vec3 _S160 = _S101 * _S131.differential_0;
	vec3 _S161 = dpsh_0.primal_0.data[8] * _S131.differential_0;
	float _S162 = _S122 * _S139 + 0.54627424478530884 * (_S161[0] + _S161[1] + _S161[2]);
	vec3 _S163 = _S98 * _S131.differential_0;
	vec3 _S164 = dpsh_0.primal_0.data[7] * _S131.differential_0;
	float s_diff_xz_T_0 = -1.09254848957061768 * (_S164[0] + _S164[1] + _S164[2]);
	vec3 _S165 = _S96 * _S131.differential_0;
	vec3 _S166 = dpsh_0.primal_0.data[6] * _S131.differential_0;
	float _S167 = 0.31539157032966614 * (_S166[0] + _S166[1] + _S166[2]);
	float _S168 = -_S167;
	vec3 _S169 = _S93 * _S131.differential_0;
	vec3 _S170 = dpsh_0.primal_0.data[5] * _S131.differential_0;
	float s_diff_yz_T_0 = -1.09254848957061768 * (_S170[0] + _S170[1] + _S170[2]);
	vec3 _S171 = _S91 * _S131.differential_0;
	vec3 _S172 = dpsh_0.primal_0.data[4] * _S131.differential_0;
	vec3 _S173 = -_S131.differential_0;
	vec3 _S174 = _S89 * _S173;
	vec3 _S175 = dpsh_0.primal_0.data[3] * _S173;
	vec3 _S176 = _S87 * _S131.differential_0;
	vec3 _S177 = dpsh_0.primal_0.data[2] * _S131.differential_0;
	vec3 _S178 = _S85 * _S131.differential_0;
	vec3 _S179 = dpsh_0.primal_0.data[1] * _S131.differential_0;
	float _S180 = 2.89061141014099121 * (z_3 * _S155) + 1.09254848957061768 * (_S172[0] + _S172[1] + _S172[2]);
	float _S181 = z_3 * (4.0 * _S151 + 2.0 * (_S146 + _S167));
	float _S182 = y_3 * (3.0 * (-_S136 + _S147) + _S152 + -_S159 + -_S162 + _S168);
	float _S183 = x_3 * (_S136 + _S152 + 3.0 * (_S147 + _S159) + _S162 + _S168);
	float _S184 = 1.44530570507049561 * (_S99 * _S139) + 0.37317633628845215 * (_S116 * _S145) + _S107 * _S155 +
	              0.48860251903533936 * (_S177[0] + _S177[1] + _S177[2]) + x_3 * s_diff_xz_T_0 + y_3 * s_diff_yz_T_0 +
	              _S181 + _S181;
	float _S185 = -0.4570457935333252 * (_S111 * _S150) + -0.59004360437393188 * (_S104 * _S158) +
	              -0.48860251903533936 * (_S179[0] + _S179[1] + _S179[2]) + z_3 * s_diff_yz_T_0 + x_3 * _S180 + _S182 +
	              _S182;
	float _S186 = -0.59004360437393188 * (_S126 * _S135) + -0.4570457935333252 * (_S111 * _S142) +
	              0.48860251903533936 * (_S175[0] + _S175[1] + _S175[2]) + z_3 * s_diff_xz_T_0 + y_3 * _S180 + _S183 +
	              _S183;
	vec3 _S187 = vec3(0.282094806432724) * _S131.differential_0;
	vec3 _S188[16];
	_S188[0] = _S130;
	_S188[1] = _S130;
	_S188[2] = _S130;
	_S188[3] = _S130;
	_S188[4] = _S130;
	_S188[5] = _S130;
	_S188[6] = _S130;
	_S188[7] = _S130;
	_S188[8] = _S130;
	_S188[9] = _S130;
	_S188[10] = _S130;
	_S188[11] = _S130;
	_S188[12] = _S130;
	_S188[13] = _S130;
	_S188[14] = _S130;
	_S188[15] = _S130;
	_S188[15] = _S133;
	_S188[14] = _S137;
	_S188[13] = _S140;
	_S188[12] = _S143;
	_S188[11] = _S148;
	_S188[10] = _S153;
	_S188[9] = _S156;
	_S188[8] = _S160;
	_S188[7] = _S163;
	_S188[6] = _S165;
	_S188[5] = _S169;
	_S188[4] = _S171;
	_S188[3] = _S174;
	_S188[2] = _S176;
	_S188[1] = _S178;
	_S188[0] = _S187;
	vec3 _S189 = vec3(_S186, _S185, _S184);
	dpdir_0.primal_0 = dpdir_0.primal_0;
	dpdir_0.differential_0 = _S189;
	SH _S190 = SH_x24_syn_dzero_0();
	_S190.data = _S188;
	dpsh_0.primal_0 = dpsh_0.primal_0;
	dpsh_0.differential_0 = _S190;
	return;
}
void s_bwd_prop_sqrt_0(inout DiffPair_float_0 _S191, float _S192) {
	_d_sqrt_0(_S191, _S192);
	return;
}
void s_bwd_prop_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_5, float _s_dOut_1) {
	float _S193 = dpx_5.primal_0[0];
	float _S194 = dpx_5.primal_0[1];
	float _S195 = dpx_5.primal_0[2];
	DiffPair_float_0 _S196;
	_S196.primal_0 = _S193 * _S193 + _S194 * _S194 + _S195 * _S195;
	_S196.differential_0 = 0.0;
	s_bwd_prop_sqrt_0(_S196, _s_dOut_1);
	float _S197 = dpx_5.primal_0[2] * _S196.differential_0;
	float _S198 = _S197 + _S197;
	float _S199 = dpx_5.primal_0[1] * _S196.differential_0;
	float _S200 = _S199 + _S199;
	float _S201 = dpx_5.primal_0[0] * _S196.differential_0;
	float _S202 = _S201 + _S201;
	vec3 _S203 = vec3(0.0);
	_S203[2] = _S198;
	_S203[1] = _S200;
	_S203[0] = _S202;
	dpx_5.primal_0 = dpx_5.primal_0;
	dpx_5.differential_0 = _S203;
	return;
}
void s_bwd_length_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S204, float _S205) {
	s_bwd_prop_length_impl_0(_S204, _S205);
	return;
}
void s_bwd_prop_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_6, vec3 _s_dOut_2) {
	float _S206 = length(dpx_6.primal_0);
	vec3 _S207 = dpx_6.primal_0 * _s_dOut_2;
	vec3 _S208 = vec3(1.0 / _S206) * _s_dOut_2;
	float _S209 = -((_S207[0] + _S207[1] + _S207[2]) / (_S206 * _S206));
	vec3 _S210 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S211;
	_S211.primal_0 = dpx_6.primal_0;
	_S211.differential_0 = _S210;
	s_bwd_length_impl_0(_S211, _S209);
	vec3 _S212 = _S208 + _S211.differential_0;
	dpx_6.primal_0 = dpx_6.primal_0;
	dpx_6.differential_0 = _S212;
	return;
}
void s_bwd_normalize_impl_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S213, vec3 _S214) {
	s_bwd_prop_normalize_impl_0(_S213, _S214);
	return;
}
void s_bwd_prop_cov2conic_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpcov_0, vec3 _s_dOut_3) {
	float _S215 = dpcov_0.primal_0.x;
	float _S216 = dpcov_0.primal_0.z;
	float _S217 = dpcov_0.primal_0.y;
	float det_0 = _S215 * _S216 - _S217 * _S217;
	vec3 _S218 = _s_dOut_3 / vec3(det_0 * det_0);
	vec3 _S219 = vec3(_S216, -_S217, _S215) * -_S218;
	vec3 _S220 = vec3(det_0) * _S218;
	float _S221 = _S219[0] + _S219[1] + _S219[2];
	float _S222 = _S217 * -_S221;
	vec3 _S223 = vec3(_S220[2] + _S216 * _S221, -_S220[1] + _S222 + _S222, _S220[0] + _S215 * _S221);
	dpcov_0.primal_0 = dpcov_0.primal_0;
	dpcov_0.differential_0 = _S223;
	return;
}
void s_bwd_prop_mul_0(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S224,
                      inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S225, mat3x3 _S226) {
	mul_0(_S224, _S225, _S226);
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
	float _S227 = 2.0 * -_s_dOut_4[2][2];
	float _S228 = 2.0 * _s_dOut_4[2][1];
	float _S229 = 2.0 * _s_dOut_4[2][0];
	float _S230 = 2.0 * _s_dOut_4[1][2];
	float _S231 = _S228 + -_S230;
	float _S232 = _S228 + _S230;
	float _S233 = 2.0 * -_s_dOut_4[1][1];
	float _S234 = x_4 * (_S227 + _S233);
	float _S235 = 2.0 * _s_dOut_4[1][0];
	float _S236 = 2.0 * _s_dOut_4[0][2];
	float _S237 = -_S229 + _S236;
	float _S238 = _S229 + _S236;
	float _S239 = 2.0 * _s_dOut_4[0][1];
	float _S240 = _S235 + -_S239;
	float _S241 = _S235 + _S239;
	float _S242 = 2.0 * -_s_dOut_4[0][0];
	float _S243 = z_4 * (_S233 + _S242);
	float _S244 = y_4 * (_S227 + _S242);
	vec4 _S245 = vec4(w_2 * _S231 + _S234 + _S234 + z_4 * _S238 + y_4 * _S241,
	                  z_4 * _S232 + w_2 * _S237 + x_4 * _S241 + _S244 + _S244,
	                  y_4 * _S232 + x_4 * _S238 + w_2 * _S240 + _S243 + _S243, x_4 * _S231 + y_4 * _S237 + z_4 * _S240);
	dpquat_1.primal_0 = dpquat_1.primal_0;
	dpquat_1.differential_0 = _S245;
	return;
}
void s_bwd_prop_scale2matrix_0(inout DiffPair_vectorx3Cfloatx2C3x3E_0 dpscale_1, mat3x3 _s_dOut_5) {
	vec3 _S246 = vec3(_s_dOut_5[0][0], _s_dOut_5[1][1], _s_dOut_5[2][2]);
	dpscale_1.primal_0 = dpscale_1.primal_0;
	dpscale_1.differential_0 = _S246;
	return;
}
void s_bwd_prop_clamp_0(inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S247, inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S248,
                        inout DiffPair_vectorx3Cfloatx2C2x3E_0 _S249, vec2 _S250) {
	_d_clamp_vector_0(_S247, _S248, _S249, _S250);
	return;
}
void s_bwd_prop_mul_1(inout DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S251, inout DiffPair_vectorx3Cfloatx2C3x3E_0 _S252,
                      vec3 _S253) {
	_d_mul_0(_S251, _S252, _S253);
	return;
}
void s_bwd_prop_splat2splatView_0(inout DiffPair_Splat_0 dpsplat_0, vec3 camPos_1, mat3x3 camViewMat_1, vec2 camFocal_1,
                                  vec2 camResolution_1, SplatView _s_dOut_6) {
	vec3 camMean_1 = dpsplat_0.primal_0.geom.mean - camPos_1;
	vec3 _S254 = s_primal_ctx_mul_0(camViewMat_1, camMean_1);
	float _S255 = _S254.z;
	float invViewMeanZ_1 = 1.0 / _S255;
	vec2 _S256 = vec2(invViewMeanZ_1);
	float _S257 = _S255 * _S255;
	vec2 _S258 = _S254.xy * camFocal_1;
	vec2 projMean_1 = _S258 * invViewMeanZ_1;
	vec2 camHalfRes_1 = camResolution_1 * 0.5;
	vec2 _S259 = -1.29999995231628418 * camHalfRes_1;
	vec2 _S260 = 1.29999995231628418 * camHalfRes_1;
	vec2 _S261 = s_primal_ctx_clamp_0(projMean_1, _S259, _S260);
	mat3x3 _S262 = s_primal_ctx_scale2matrix_0(dpsplat_0.primal_0.geom.scale);
	mat3x3 _S263 = s_primal_ctx_quat2matrix_0(dpsplat_0.primal_0.geom.quat);
	float _S264 = camFocal_1.x;
	float _S265 = -_S261.x;
	float _S266 = camFocal_1.y;
	float _S267 = -_S261.y;
	mat3x3 J_0 = mat3x3(vec3(_S264 * invViewMeanZ_1, 0.0, _S265 * invViewMeanZ_1),
	                    vec3(0.0, _S266 * invViewMeanZ_1, _S267 * invViewMeanZ_1), vec3(0.0, 0.0, 0.0));
	mat3x3 _S268 = s_primal_ctx_mul_1(J_0, camViewMat_1);
	mat3x3 _S269 = s_primal_ctx_mul_1(_S263, _S262);
	mat3x3 _S270 = s_primal_ctx_mul_1(_S268, _S269);
	mat3x3 _S271 = transpose(_S270);
	mat3x3 _S272 = s_primal_ctx_mul_1(_S270, _S271);
	mat2x2 _S273 = mat2x2(_S272[0].xy, _S272[1].xy);
	vec3 cov2D_1 = vec3(_S273[0][0], _S273[0][1], _S273[1][1]) + vec3(0.30000001192092896, 0.0, 0.30000001192092896);
	vec3 _S274 = normalize(camMean_1);
	SH _S275 = SH_x24_syn_dzero_0();
	DiffPair_SH_0 _S276;
	_S276.primal_0 = dpsplat_0.primal_0.sh;
	_S276.differential_0 = _S275;
	vec3 _S277 = vec3(0.0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S278;
	_S278.primal_0 = _S274;
	_S278.differential_0 = _S277;
	s_bwd_prop_sh2color_0(_S276, _S278, _s_dOut_6.color);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S279;
	_S279.primal_0 = camMean_1;
	_S279.differential_0 = _S277;
	s_bwd_normalize_impl_0(_S279, _S278.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S280;
	_S280.primal_0 = cov2D_1;
	_S280.differential_0 = _S277;
	s_bwd_prop_cov2conic_0(_S280, _s_dOut_6.geom.conic);
	vec2 _S281 = vec2(0.0);
	vec2 _S282 = _S281;
	_S282[1] = _S280.differential_0[2];
	vec2 _S283 = _S281;
	_S283[1] = _S280.differential_0[1];
	_S283[0] = _S280.differential_0[0];
	mat2x2 _S284 = mat2x2(0.0, 0.0, 0.0, 0.0);
	_S284[1] = _S282;
	_S284[0] = _S283;
	vec3 _S285 = vec3(_S284[1][0], _S284[1][1], 0.0);
	vec3 _S286 = vec3(_S284[0][0], _S284[0][1], 0.0);
	mat3x3 _S287 = mat3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	mat3x3 _S288 = _S287;
	_S288[1] = _S285;
	_S288[0] = _S286;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S289;
	_S289.primal_0 = _S270;
	_S289.differential_0 = _S287;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S290;
	_S290.primal_0 = _S271;
	_S290.differential_0 = _S287;
	s_bwd_prop_mul_0(_S289, _S290, _S288);
	mat3x3 _S291 = _S289.differential_0 + transpose(_S290.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S292;
	_S292.primal_0 = _S268;
	_S292.differential_0 = _S287;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S293;
	_S293.primal_0 = _S269;
	_S293.differential_0 = _S287;
	s_bwd_prop_mul_0(_S292, _S293, _S291);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S294;
	_S294.primal_0 = _S263;
	_S294.differential_0 = _S287;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S295;
	_S295.primal_0 = _S262;
	_S295.differential_0 = _S287;
	s_bwd_prop_mul_0(_S294, _S295, _S293.differential_0);
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S296;
	_S296.primal_0 = J_0;
	_S296.differential_0 = _S287;
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S297;
	_S297.primal_0 = camViewMat_1;
	_S297.differential_0 = _S287;
	s_bwd_prop_mul_0(_S296, _S297, _S292.differential_0);
	float _S298 = _S267 * _S296.differential_0[1][2];
	float _S299 = -(invViewMeanZ_1 * _S296.differential_0[1][2]);
	float _S300 = _S266 * _S296.differential_0[1][1];
	float _S301 = _S265 * _S296.differential_0[0][2];
	float _S302 = -(invViewMeanZ_1 * _S296.differential_0[0][2]);
	float _S303 = _S264 * _S296.differential_0[0][0];
	vec4 _S304 = vec4(0.0);
	DiffPair_vectorx3Cfloatx2C4x3E_0 _S305;
	_S305.primal_0 = dpsplat_0.primal_0.geom.quat;
	_S305.differential_0 = _S304;
	s_bwd_prop_quat2matrix_0(_S305, _S294.differential_0);
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S306;
	_S306.primal_0 = dpsplat_0.primal_0.geom.scale;
	_S306.differential_0 = _S277;
	s_bwd_prop_scale2matrix_0(_S306, _S295.differential_0);
	vec2 _S307 = vec2(_S302, _S299);
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S308;
	_S308.primal_0 = projMean_1;
	_S308.differential_0 = _S281;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S309;
	_S309.primal_0 = _S259;
	_S309.differential_0 = _S281;
	DiffPair_vectorx3Cfloatx2C2x3E_0 _S310;
	_S310.primal_0 = _S260;
	_S310.differential_0 = _S281;
	s_bwd_prop_clamp_0(_S308, _S309, _S310, _S307);
	vec2 _S311 = _s_dOut_6.geom.mean2D + _S308.differential_0;
	vec2 _S312 = _S258 * _S311;
	vec2 _S313 = camFocal_1 * (_S256 * _S311);
	vec3 _S314 = vec3(_S313[0], _S313[1], -((_S298 + _S300 + _S301 + _S303 + _S312[0] + _S312[1]) / _S257));
	DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S315;
	_S315.primal_0 = camViewMat_1;
	_S315.differential_0 = _S287;
	DiffPair_vectorx3Cfloatx2C3x3E_0 _S316;
	_S316.primal_0 = camMean_1;
	_S316.differential_0 = _S277;
	s_bwd_prop_mul_1(_S315, _S316, _S314);
	vec3 _S317 = _S279.differential_0 + _S316.differential_0;
	SplatGeom _S318 = SplatGeom_x24_syn_dzero_0();
	_S318.opacity = _s_dOut_6.geom.opacity;
	_S318.quat = _S305.differential_0;
	_S318.scale = _S306.differential_0;
	_S318.mean = _S317;
	Splat _S319 = Splat_x24_syn_dzero_0();
	_S319.sh = _S276.differential_0;
	_S319.geom = _S318;
	dpsplat_0.primal_0 = dpsplat_0.primal_0;
	dpsplat_0.differential_0 = _S319;
	return;
}
void s_bwd_splat2splatView_0(inout DiffPair_Splat_0 _S320, vec3 _S321, mat3x3 _S322, vec2 _S323, vec2 _S324,
                             SplatView _S325) {
	s_bwd_prop_splat2splatView_0(_S320, _S321, _S322, _S323, _S324, _S325);
	return;
}
Splat bwd_splat2splatView(Splat splat_1, vec3 camPos_2, mat3x3 camViewMat_2, vec2 camFocal_2, vec2 camResolution_2,
                          SplatView dL_dsplatView_0) {
	Splat _S326 = Splat_x24_syn_dzero_0();
	DiffPair_Splat_0 dp_0;
	dp_0.primal_0 = splat_1;
	dp_0.differential_0 = _S326;
	s_bwd_splat2splatView_0(dp_0, camPos_2, camViewMat_2, camFocal_2, camResolution_2, dL_dsplatView_0);
	return dp_0.differential_0;
}
void _d_exp_0(inout DiffPair_float_0 dpx_7, float dOut_7) {
	float _S327 = exp(dpx_7.primal_0) * dOut_7;
	dpx_7.primal_0 = dpx_7.primal_0;
	dpx_7.differential_0 = _S327;
	return;
}
float splatViewGeom2alpha(SplatViewGeom splatViewGeom_0, vec2 fragCoord_0, float camHeight_0) {
	vec2 x2D_0 = vec2(fragCoord_0.x, camHeight_0 - fragCoord_0.y) - splatViewGeom_0.mean2D;
	float _S328 = x2D_0.x;
	float _S329 = x2D_0.y;
	return splatViewGeom_0.opacity *
	       exp(-0.5 * (splatViewGeom_0.conic.x * _S328 * _S328 + splatViewGeom_0.conic.z * _S329 * _S329) -
	           splatViewGeom_0.conic.y * _S328 * _S329);
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
float s_primal_ctx_exp_0(float _S330) { return exp(_S330); }
void s_bwd_prop_exp_0(inout DiffPair_float_0 _S331, float _S332) {
	_d_exp_0(_S331, _S332);
	return;
}
void s_bwd_prop_splatViewGeom2alpha_0(inout DiffPair_SplatViewGeom_0 dpsplatViewGeom_0, vec2 fragCoord_1,
                                      float camHeight_1, float _s_dOut_7) {
	vec2 x2D_1 = vec2(fragCoord_1.x, camHeight_1 - fragCoord_1.y) - dpsplatViewGeom_0.primal_0.mean2D;
	float _S333 = dpsplatViewGeom_0.primal_0.conic.x;
	float _S334 = x2D_1.x;
	float _S335 = _S333 * _S334;
	float _S336 = dpsplatViewGeom_0.primal_0.conic.z;
	float _S337 = x2D_1.y;
	float _S338 = _S336 * _S337;
	float _S339 = dpsplatViewGeom_0.primal_0.conic.y;
	float _S340 = _S339 * _S334;
	float power_0 = -0.5 * (_S335 * _S334 + _S338 * _S337) - _S340 * _S337;
	float _S341 = dpsplatViewGeom_0.primal_0.opacity * _s_dOut_7;
	float _S342 = s_primal_ctx_exp_0(power_0) * _s_dOut_7;
	DiffPair_float_0 _S343;
	_S343.primal_0 = power_0;
	_S343.differential_0 = 0.0;
	s_bwd_prop_exp_0(_S343, _S341);
	float _S344 = -_S343.differential_0;
	float _S345 = _S337 * _S344;
	float _S346 = -0.5 * _S343.differential_0;
	float _S347 = _S337 * _S346;
	float _S348 = _S334 * _S346;
	vec3 _S349 = vec3(_S334 * _S348, _S334 * _S345, _S337 * _S347);
	vec2 _S350 = -vec2(_S339 * _S345 + _S335 * _S346 + _S333 * _S348, _S340 * _S344 + _S338 * _S346 + _S336 * _S347);
	SplatViewGeom _S351 = SplatViewGeom_x24_syn_dzero_0();
	_S351.opacity = _S342;
	_S351.conic = _S349;
	_S351.mean2D = _S350;
	dpsplatViewGeom_0.primal_0 = dpsplatViewGeom_0.primal_0;
	dpsplatViewGeom_0.differential_0 = _S351;
	return;
}
void s_bwd_splatViewGeom2alpha_0(inout DiffPair_SplatViewGeom_0 _S352, vec2 _S353, float _S354, float _S355) {
	s_bwd_prop_splatViewGeom2alpha_0(_S352, _S353, _S354, _S355);
	return;
}
SplatViewGeom bwd_splatViewGeom2alpha(SplatViewGeom splatViewGeom_1, vec2 fragCoord_2, float camHeight_2,
                                      float dL_dalpha_0) {
	SplatViewGeom _S356 = SplatViewGeom_x24_syn_dzero_0();
	DiffPair_SplatViewGeom_0 dp_1;
	dp_1.primal_0 = splatViewGeom_1;
	dp_1.differential_0 = _S356;
	s_bwd_splatViewGeom2alpha_0(dp_1, fragCoord_2, camHeight_2, dL_dalpha_0);
	return dp_1.differential_0;
}
#endif
