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
static int argmax(const float *v, int n) {
  int idx = 0; float best = v[0];
  for (int i = 1; i < n; ++i) if (v[i] > best) { best = v[i]; idx = i; }
  return idx;
}
static void mlp_forward_logits(const float *x_raw, float *z2_out) {
  static float x[4096];   // buffers simples; cabe folgado para este MLP
  static float z1[2048];
  static float h[2048];

  for (int i = 0; i < MLP_N_IN; ++i) x[i] = x_raw[i];
  normalize_inplace(x, MLP_N_IN);

  // z1 = x*W1 + b1
  matmul_vec(x, MLP_W1, MLP_N_IN, MLP_N_HID, z1);
  for (int j = 0; j < MLP_N_HID; ++j) h[j] = relu(z1[j] + MLP_B1[j]);

  // z2 = h*W2 + b2
  matmul_vec(h, MLP_W2, MLP_N_HID, MLP_N_OUT, z2_out);
  for (int k = 0; k < MLP_N_OUT; ++k) z2_out[k] += MLP_B2[k];
}