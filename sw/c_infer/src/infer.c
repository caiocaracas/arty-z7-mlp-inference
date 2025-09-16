#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "../export/weights.h" 

static inline float relu(float x) {return x > 0.f ? x : 0.f; }

static void normalize_inplace(float *x, int d) {
    for (int i = 0; i < d; ++i) {
        x[i] = (x[i] - MLP_FEAT_MEAN[i]) / (MLP_FEAT_SCALE[i] + 1e-12f);
    }
}
static void matmul_vec(const float *x, const float *W, int d, int m, float *y) {
  for (int j = 0; j < m; ++j) {
    float acc = 0.f;
    const float *wcol = &W[j];    
    for (int i = 0; i < d; ++i) acc += x[i] * wcol[i * m];
    y[j] = acc;
  }
}
