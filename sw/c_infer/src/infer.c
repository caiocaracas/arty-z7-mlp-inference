#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <stdint.h>
#include <time.h>
#include "../include/weights.h"  

// utilitários numéricos
static inline float relu(float x) { return x > 0.f ? x : 0.f; }

static void normalize_inplace(float *x, int d) {
  for (int i = 0; i < d; ++i)
    x[i] = (x[i] - MLP_FEAT_MEAN[i]) / (MLP_FEAT_SCALE[i] + 1e-12f);
}

// y = x(1xd) * W(dxm)  com W em row-major (d x m)
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
  static float x[4096], z1[2048], h[2048];
  for (int i = 0; i < MLP_N_IN; ++i) x[i] = x_raw[i];
  normalize_inplace(x, MLP_N_IN);
  matmul_vec(x, MLP_W1, MLP_N_IN, MLP_N_HID, z1);
  for (int j = 0; j < MLP_N_HID; ++j) h[j] = relu(z1[j] + MLP_B1[j]);
  matmul_vec(h, MLP_W2, MLP_N_HID, MLP_N_OUT, z2_out);
  for (int k = 0; k < MLP_N_OUT; ++k) z2_out[k] += MLP_B2[k];
}

// helpers de JSON/arquivo 

static int first_non_ws(FILE *fp) {
  int c; do { c = fgetc(fp); } while (c != EOF && isspace(c));
  return c;
}

static char* slurp_file(const char *path, long *out_sz) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;
  fseek(fp, 0, SEEK_END);
  long sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *buf = (char*)malloc((size_t)sz + 1);
  if (!buf) { fclose(fp); return NULL; }
  if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) { free(buf); fclose(fp); return NULL; }
  buf[sz] = '\0'; fclose(fp);
  if (out_sz) *out_sz = sz;
  return buf;
}

// timing & estatísticas
#ifndef INFER_REPEAT // número de repetições de forward por amostra para reduzir quantização do relógio
#define INFER_REPEAT 5000
#endif

static inline uint64_t now_ns(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (uint64_t)tv.tv_sec * 1000000000ull + (uint64_t)tv.tv_usec * 1000ull;
}

static int cmp_double(const void *a, const void *b) {
  double da = *(const double*)a, db = *(const double*)b;
  return (da > db) - (da < db);
}

static double percentile_sorted(const double *v, int n, double p) {
  if (n <= 0) return 0.0;
  double idx = (p/100.0) * (n - 1);
  int i = (int)idx;
  return v[i];
}

// constroi caminho "<dir_do_json>/results_batch.json" em out_path
static void make_results_path(const char *json_path, char *out_path, size_t cap) {
  const char *slash = strrchr(json_path, '/');
  if (!slash) {
    snprintf(out_path, cap, "results_batch.json");
  } else {
    size_t dirlen = (size_t)(slash - json_path);
    if (dirlen + 1 + strlen("results_batch.json") + 1 > cap) {
      snprintf(out_path, cap, "results_batch.json");
      return;
    }
    memcpy(out_path, json_path, dirlen);
    out_path[dirlen] = '\0';
    strcat(out_path, "/results_batch.json");
  }
}

// SINGLE: {"x_raw":[...], "pred_argmax_logits":K}

static int load_x_raw_from_json_single(const char *path, float *x, int n, int *pred_json_opt) {
  long sz=0; char *buf = slurp_file(path, &sz);
  if (!buf) { perror("fopen"); return -1; }

  const char *key = "\"x_raw\"";
  char *p = strstr(buf, key);
  if (!p) { free(buf); return -2; }
  p = strchr(p, '['); if (!p) { free(buf); return -3; }
  ++p;

  for (int i = 0; i < n; ++i) {
    char *endp = NULL;
    x[i] = strtof(p, &endp);
    if (endp == p) { free(buf); return -4; }
    p = endp;
    while (*p && *p != ']' && *p != '-' && *p != '+' && *p != '.' && (*p < '0' || *p > '9')) ++p;
  }

  if (pred_json_opt) {
    const char *k2 = "\"pred_argmax_logits\"";
    char *q = strstr(p, k2);
    if (q) {
      q = strchr(q, ':'); if (q) *pred_json_opt = (int)strtol(q+1, NULL, 10);
    } else {
      *pred_json_opt = -1;
    }
  }

  free(buf);
  return 0;
}

static int run_single(const char *path) {
  float x_raw[MLP_N_IN], z2[MLP_N_OUT];
  int pred_json = -1;

  if (load_x_raw_from_json_single(path, x_raw, MLP_N_IN, &pred_json) != 0) {
    fprintf(stderr, "Erro ao ler %s\n", path);
    return 1;
  }
  mlp_forward_logits(x_raw, z2);
  int pred_c = argmax(z2, MLP_N_OUT);

  printf("Pred(JSON esperado) = %d\n", pred_json);
  printf("Pred(C calculado)   = %d\n", pred_c);
  printf("Logits:");
  for (int k = 0; k < MLP_N_OUT; ++k) printf(" %d:%.6f", k, z2[k]);
  printf("\n");
  return (pred_json >= 0 && pred_c != pred_json) ? 2 : 0;
}

// BATCH: [{"x_raw":[...], "pred":K, "margin":M}, ...]

static int run_batch(const char *path) {
  long sz=0; char *buf = slurp_file(path, &sz);
  if (!buf) { perror("fopen"); return 1; }

  const char *p = buf;
  const char *key = "\"x_raw\"";
  int n = 0, hits = 0;
  float x_raw[MLP_N_IN], z2[MLP_N_OUT];

  // 1º passe: contar amostras
  const char *scan = buf;
  while ((scan = strstr(scan, key)) != NULL) { ++n; scan += 6; }
  if (n == 0) { free(buf); fprintf(stderr,"Nenhuma amostra em %s\n", path); return 5; }

  // arrays: tempos (µs por amostra) e margens (zmax - z2nd)
  double *t_us = (double*)malloc(sizeof(double)*n);
  double *margins = (double*)malloc(sizeof(double)*n);
  if (!t_us) { free(buf); return 6; }
  if (!margins) { free(t_us); free(buf); return 6; }

  // warm-up (até 3 amostras)
  p = buf;
  for (int w = 0; w < 3 && strstr(p, key); ++w) {
    const char *arr = strstr(p, key); arr = strchr(arr, '['); if (!arr) break; ++arr;
    for (int i = 0; i < MLP_N_IN; ++i) {
      char *endp = NULL;
      x_raw[i] = strtof(arr, &endp);
      if (endp == arr) { free(margins); free(t_us); free(buf); return 2; }
      arr = endp;
      while (*arr && *arr != ']' && *arr != '-' && *arr != '+' && *arr != '.' && (*arr < '0' || *arr > '9')) ++arr;
    }
    mlp_forward_logits(x_raw, z2);
    p = arr;
  }

  // loop medido — também acumulamos tempo total do lote
  p = buf;
  int idx = 0;
  uint64_t t_batch0 = now_ns();
  while ((p = strstr(p, key)) != NULL && idx < n) {
    const char *arr = strchr(p, '['); if (!arr) break; ++arr;

    for (int i = 0; i < MLP_N_IN; ++i) {
      char *endp = NULL;
      x_raw[i] = strtof(arr, &endp);
      if (endp == arr) { free(margins); free(t_us); free(buf); return 2; }
      arr = endp;
      while (*arr && *arr != ']' && *arr != '-' && *arr != '+' && *arr != '.' && (*arr < '0' || *arr > '9')) ++arr;
    }

    const char *q = strstr(arr, "\"pred\"");
    if (!q) { free(margins); free(t_us); free(buf); return 3; }
    q = strchr(q, ':'); if (!q) { free(margins); free(t_us); free(buf); return 4; }
    int pred_json = (int)strtol(q+1, NULL, 10);

    // mede INFER_REPEAT forward passes e divide depois (tempo por amostra)
    uint64_t t0 = now_ns();
    for (int rep = 0; rep < INFER_REPEAT; ++rep) {
      mlp_forward_logits(x_raw, z2);
    }
    uint64_t t1 = now_ns();

    int pred_c = argmax(z2, MLP_N_OUT);
    if (pred_c == pred_json) ++hits;

    t_us[idx] = ((double)(t1 - t0) / 1000.0) / (double)INFER_REPEAT; // µs por inferência

    // margem: maior - segundo_maior (sem softmax)
    double max1 = z2[0], max2 = -1e30;
    for (int k = 1; k < MLP_N_OUT; ++k) {
      double v = z2[k];
      if (v > max1) { max2 = max1; max1 = v; }
      else if (v > max2) { max2 = v; }
    }
    margins[idx] = max1 - max2;

    ++idx;
    p = arr; // avança
  }
  uint64_t t_batch1 = now_ns();
  double mean_us_total = ((double)(t_batch1 - t_batch0) / 1000.0) / ((double)idx * (double)INFER_REPEAT);

  // estatística de tempo por amostra (apenas para p95)
  qsort(t_us, idx, sizeof(double), cmp_double);
  double p95_us = percentile_sorted(t_us, idx, 95.0);

  // estatística de margem
  qsort(margins, idx, sizeof(double), cmp_double);
  double margin_p10 = percentile_sorted(margins, idx, 10.0);
  double margin_p50 = percentile_sorted(margins, idx, 50.0);

  double parity = 100.0 * (double)hits / (double)idx;
  printf("Batch: N=%d  acertos=%d  PARIDADE_LOTE=%.2f%%  mean=%.2f us/inf  p95=%.2f us/inf  "
         "margin_p10=%.6f  margin_p50=%.6f\n",
         idx, hits, parity, mean_us_total, p95_us, margin_p10, margin_p50);

  // grava results_batch.json no mesmo diretório do JSON de entrada
  char outp[1024];
  make_results_path(path, outp, sizeof(outp));
  FILE *fr = fopen(outp, "wb");
  if (fr) {
    time_t ts = time(NULL);
    fprintf(fr,
      "{\n"
      "  \"n\": %d,\n"
      "  \"parity_batch_pct\": %.6f,\n"
      "  \"infer_mean_us\": %.6f,\n"
      "  \"infer_p95_us\": %.6f,\n"
      "  \"margin_p10\": %.6f,\n"
      "  \"margin_p50\": %.6f,\n"
      "  \"repeat\": %d,\n"
      "  \"timestamp\": %ld\n"
      "}\n",
      idx, parity, mean_us_total, p95_us, margin_p10, margin_p50, INFER_REPEAT, (long)ts);
    fclose(fr);
    printf("[OK] results_batch.json escrito em: %s\n", outp);
  } else {
    perror("fopen results_batch.json");
  }

  free(t_us);
  free(margins);
  free(buf);
  return (hits == idx) ? 0 : 2;
}

// entrada/saída

static const char* resolve_default_json(void) {
  // tenta primeiro batch; se não houver, cai para single
  static const char *cands[] = {
    // batch
    "export/test_batch.json",
    "../export/test_batch.json",
    "../../export/test_batch.json",
    "../../../export/test_batch.json",
     "ml/src/export/test_batch.json",
    "../ml/src/export/test_batch.json",
    "../../ml/src/export/test_batch.json",
    "../../../ml/src/export/test_batch.json",
    // single
    "export/test_vector.json",
    "../export/test_vector.json",
    "../../export/test_vector.json",
    "../../../export/test_vector.json",
     "ml/src/export/test_vector.json",
    "../ml/src/export/test_vector.json",
    "../../ml/src/export/test_vector.json",
    "../../../ml/src/export/test_vector.json",
    NULL
  };
  for (int i = 0; cands[i]; ++i) {
    FILE *fp = fopen(cands[i], "rb");
    if (fp) { fclose(fp); return cands[i]; }
  }
  return NULL;
}

int main(int argc, char **argv) {
  const char *json_path = (argc > 1) ? argv[1] : resolve_default_json();
  if (!json_path) {
    fprintf(stderr, "Não achei test_batch.json nem test_vector.json.\n");
    return 1;
  }

  FILE *fp = fopen(json_path, "rb");
  if (!fp) { perror("fopen"); fprintf(stderr, "Falha abrindo %s\n", json_path); return 1; }
  int c = first_non_ws(fp);
  fclose(fp);

  if (c == '[')      return run_batch(json_path);
  else if (c == '{') return run_single(json_path);
  else { fprintf(stderr, "JSON inválido em %s\n", json_path); return 1; }
}