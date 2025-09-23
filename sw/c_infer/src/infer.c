//------------------------------------------------------------------------------
// infer.c
//
// Inferência MLP (float32 + ReLU) e métricas estáveis sobre lote.
// Lê export/test_batch.json (ou test_vector.json) e grava export/results_batch.json.
//
// Compilação típica:
//   cc -O3 -std=c11 -Wall -Wextra -o infer infer.c
//
// Uso:
//   ./infer                # busca export/test_batch.json por padrão
//   ./infer caminho.json   # pode apontar para test_vector.json
//------------------------------------------------------------------------------

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <ctype.h>

#include "../include/weights.h"  // Gerado por train.py

//==============================================================================
// Utilitários numéricos e de tempo
//==============================================================================

static inline float Relu(float x) { return x > 0.0f ? x : 0.0f; }

static inline double NowUs(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

static inline void NormalizeInPlace(float* x, int d) {
  for (int i = 0; i < d; ++i) {
    x[i] = (x[i] - MLP_FEAT_MEAN[i]) / (MLP_FEAT_SCALE[i] + 1e-12f);
  }
}

static inline void DenseRelu(const float* w_row_major, const float* b,
                             const float* x, int in, int out, float* y) {
  // y[j] = relu( sum_i w[j,i] * x[i] + b[j] ).
  for (int j = 0; j < out; ++j) {
    const float* w = w_row_major + (size_t)j * (size_t)in;
    double acc = (double)b[j];
    for (int i = 0; i < in; ++i) {
      acc += (double)w[i] * (double)x[i];
    }
    y[j] = Relu((float)acc);
  }
}

static inline void DenseLinear(const float* w_row_major, const float* b,
                               const float* x, int in, int out, float* y) {
  // y[j] = sum_i w[j,i] * x[i] + b[j].
  for (int j = 0; j < out; ++j) {
    const float* w = w_row_major + (size_t)j * (size_t)in;
    double acc = (double)b[j];
    for (int i = 0; i < in; ++i) {
      acc += (double)w[i] * (double)x[i];
    }
    y[j] = (float)acc;
  }
}

static inline int ArgMaxF(const float* v, int n) {
  int a = 0;
  float m = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > m) {
      m = v[i];
      a = i;
    }
  }
  return a;
}

static inline void Top2Margin(const float* v, int n, float* margin_out) {
  // Encontra top1 e top2 em uma única varredura.
  float t1 = -INFINITY, t2 = -INFINITY;
  for (int i = 0; i < n; ++i) {
    const float z = v[i];
    if (z >= t1) {
      t2 = t1;
      t1 = z;
    } else if (z > t2) {
      t2 = z;
    }
  }
  *margin_out = t1 - t2;
}

static inline double LogSumExpF(const float* v, int n) {
  float m = v[0];
  for (int i = 1; i < n; ++i) {
    if (v[i] > m) m = v[i];
  }
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    s += exp((double)v[i] - (double)m);
  }
  return (double)m + log(s + 1e-300);
}

// Percentis p∈[0,100] de vetor já ordenado (asc).
static int CompareDouble(const void* a, const void* b) {
  const double da = *(const double*)a;
  const double db = *(const double*)b;
  if (da < db) return -1;
  if (da > db) return +1;
  return 0;
}

static double PercentileSorted(const double* x_sorted, int n, double p) {
  if (n <= 0) return 0.0;
  if (p <= 0.0) return x_sorted[0];
  if (p >= 100.0) return x_sorted[n - 1];
  const double pos = (p / 100.0) * (n - 1);
  const int i = (int)pos;
  const double frac = pos - i;
  if (i + 1 < n) {
    return x_sorted[i] * (1.0 - frac) + x_sorted[i + 1] * frac;
  }
  return x_sorted[i];
}

//==============================================================================
// JSON helpers (parsing leve, suficiente para o formato)
//==============================================================================

static int FirstNonWs(FILE* fp) {
  int c;
  do {
    c = fgetc(fp);
  } while (c != EOF && isspace(c));
  return c;
}

static char* SlurpFile(const char* path, size_t* len_out) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long L = ftell(f);
  if (L < 0) {
    fclose(f);
    return NULL;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return NULL;
  }
  char* buf = (char*)malloc((size_t)L + 1u);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  if (fread(buf, 1u, (size_t)L, f) != (size_t)L) {
    fclose(f);
    free(buf);
    return NULL;
  }
  buf[L] = '\0';
  fclose(f);
  if (len_out) *len_out = (size_t)L;
  return buf;
}

// Extrai "x_raw":[...], "pred_py":int (opcional), "label":int.
static int ParseItemFields(const char* js, float* x_out, int dx,
                           int* pred_py_opt, int* label_opt) {
  const char* px = strstr(js, "\"x_raw\"");
  if (!px) return -1;
  const char* qb = strchr(px, '[');
  const char* qe = qb ? strchr(qb, ']') : NULL;
  if (!qb || !qe) return -1;

  int i = 0;
  const char* p = qb + 1;
  while (p < qe && i < dx) {
    while (p < qe && isspace((unsigned char)*p)) ++p;
    x_out[i++] = strtof(p, (char**)&p);
    const char* c = strchr(p, ',');
    if (c && c < qe) {
      p = c + 1;
    } else {
      break;
    }
  }
  if (i != dx) return -2;

  if (pred_py_opt) {
    const char* pp = strstr(js, "\"pred_py\"");
    if (pp) {
      pp = strchr(pp, ':');
      if (pp) *pred_py_opt = (int)strtol(pp + 1, NULL, 10);
    } else {
      *pred_py_opt = -1;
    }
  }
  if (label_opt) {
    const char* pl = strstr(js, "\"label\"");
    if (pl) {
      pl = strchr(pl, ':');
      if (pl) *label_opt = (int)strtol(pl + 1, NULL, 10);
    } else {
      *label_opt = -1;
    }
  }
  return 0;
}

//==============================================================================
// Execução (single e batch)
//==============================================================================

static void MlpInferOne(const float* x_raw, float* z2_out) {
  // Normaliza -> densa+ReLU -> densa linear.
  float x[MLP_N_IN];
  for (int i = 0; i < MLP_N_IN; ++i) x[i] = x_raw[i];

  NormalizeInPlace(x, MLP_N_IN);

  float h[MLP_N_HID];
  DenseRelu(MLP_W1, MLP_B1, x, MLP_N_IN, MLP_N_HID, h);

  DenseLinear(MLP_W2, MLP_B2, h, MLP_N_HID, MLP_N_OUT, z2_out);
}

static int RunSingle(const char* path) {
  // Lê test_vector.json e escreve results_batch.json equivalente (forwards=1).
  char* buf = SlurpFile(path, NULL);
  if (!buf) {
    perror("SlurpFile");
    return 1;
  }

  float x_raw[MLP_N_IN], z2[MLP_N_OUT];
  int pred_json = -1, label = -1;
  if (ParseItemFields(buf, x_raw, MLP_N_IN, &pred_json, &label) != 0) {
    fprintf(stderr, "Falha ao ler x_raw em %s\n", path);
    free(buf);
    return 1;
  }

  const double t0 = NowUs();
  MlpInferOne(x_raw, z2);
  const double t1 = NowUs();

  const int pred_c = ArgMaxF(z2, MLP_N_OUT);
  float margin = 0.0f;
  Top2Margin(z2, MLP_N_OUT, &margin);

  const double elapsed_us_total = (t1 - t0);
  const long long macs_per_inf =
      (long long)MLP_N_IN * (long long)MLP_N_HID +
      (long long)MLP_N_HID * (long long)MLP_N_OUT;
  const double ns_per_mac = (macs_per_inf > 0)
                                ? (elapsed_us_total * 1000.0) /
                                      (double)macs_per_inf
                                : 0.0;

  const double parity_py =
      (pred_json >= 0 && pred_json == pred_c) ? 100.0 : 0.0;

  FILE* fr = fopen("export/results_batch.json", "wb");
  if (!fr) {
    perror("open results");
    free(buf);
    return 1;
  }
  fprintf(fr,
          "{\n"
          "  \"forwards\": 1,\n"
          "  \"elapsed_us_total\": %.6f,\n"
          "  \"macs_per_inf\": %lld,\n"
          "  \"ns_per_mac\": %.6f,\n"
          "  \"parity_py_pct\": %.6f,\n"
          "  \"margin_p10\": %.6f,\n"
          "  \"margin_p50\": %.6f,\n"
          "  \"timestamp\": %ld\n"
          "}\n",
          elapsed_us_total, macs_per_inf, ns_per_mac, parity_py,
          (double)margin, (double)margin, (long)time(NULL));
  fclose(fr);

  printf(
      "Totals: forwards=1  elapsed=%.3f ms  MACs/inf=%lld  ns/MAC=%.3f  "
      "parity_py=%.2f%%\n",
      elapsed_us_total / 1000.0, macs_per_inf, ns_per_mac, parity_py);

  free(buf);
  return 0;
}

static int RunBatch(const char* path) {
  // Lê test_batch.json (array de objetos) e agrega métricas.
  FILE* fp = fopen(path, "rb");
  if (!fp) {
    perror("fopen");
    return 1;
  }
  const int c0 = FirstNonWs(fp);
  fclose(fp);
  if (c0 != '[') {
    fprintf(stderr, "JSON não é um array: %s\n", path);
    return 1;
  }

  char* buf = SlurpFile(path, NULL);
  if (!buf) {
    perror("SlurpFile");
    return 1;
  }

  int idx = 0;
  int capacity = 256;

  float* margins = (float*)malloc(sizeof(float) * (size_t)capacity);
  double* t_us = (double*)malloc(sizeof(double) * (size_t)capacity);
  if (!margins || !t_us) {
    free(buf);
    free(margins);
    free(t_us);
    return 1;
  }

  unsigned* cm = (unsigned*)calloc(
      (size_t)MLP_N_OUT * (size_t)MLP_N_OUT, sizeof(unsigned));
  if (!cm) {
    free(buf);
    free(margins);
    free(t_us);
    return 1;
  }

  int hits_py = 0;
  int correct_lab = 0;
  double loss_sum = 0.0;
  double elapsed_us_total = 0.0;

  // Varredura por objetos { ... } no nível do array.
  int depth = 0;
  char* p = buf;
  while (*p) {
    if (*p == '{' && depth == 0) {
      char* start = p;
      int d = 0;
      do {
        if (*p == '{') ++d;
        else if (*p == '}') --d;
        ++p;
      } while (*p && d > 0);

      if (idx >= capacity) {
        capacity *= 2;
        margins =
            (float*)realloc(margins, sizeof(float) * (size_t)capacity);
        t_us =
            (double*)realloc(t_us, sizeof(double) * (size_t)capacity);
        if (!margins || !t_us) {
          free(buf);
          free(margins);
          free(t_us);
          free(cm);
          return 1;
        }
      }

      float x_raw[MLP_N_IN], z2[MLP_N_OUT];
      int pred_json = -1, label = -1;
      if (ParseItemFields(start, x_raw, MLP_N_IN, &pred_json, &label) != 0) {
        fprintf(stderr, "Item %d inválido.\n", idx);
        continue;
      }

      const double t0 = NowUs();
      MlpInferOne(x_raw, z2);
      const double t1 = NowUs();
      t_us[idx] = (t1 - t0);

      const int pred_c = ArgMaxF(z2, MLP_N_OUT);
      if (pred_json >= 0 && pred_c == pred_json) ++hits_py;

      float m = 0.0f;
      Top2Margin(z2, MLP_N_OUT, &m);
      margins[idx] = m;

      if (label >= 0) {
        if (pred_c == label) ++correct_lab;
        const double lse = LogSumExpF(z2, MLP_N_OUT);
        loss_sum += (lse - (double)z2[label]);  // NLL
        cm[label * MLP_N_OUT + pred_c] += 1u;
      }

      ++idx;
      continue;
    }
    if (*p == '[') ++depth;
    else if (*p == ']') --depth;
    ++p;
  }

  for (int i = 0; i < idx; ++i) elapsed_us_total += t_us[i];

  const long long macs_per_inf =
      (long long)MLP_N_IN * (long long)MLP_N_HID +
      (long long)MLP_N_HID * (long long)MLP_N_OUT;
  const double ns_per_mac =
      (idx > 0 && macs_per_inf > 0)
          ? (elapsed_us_total * 1000.0) /
                ((double)macs_per_inf * (double)idx)
          : 0.0;

  // Percentis das margens.
  double* m_sorted = (double*)malloc(sizeof(double) * (size_t)idx);
  if (!m_sorted) {
    free(buf);
    free(margins);
    free(t_us);
    free(cm);
    return 1;
  }
  for (int i = 0; i < idx; ++i) m_sorted[i] = (double)margins[i];
  qsort(m_sorted, (size_t)idx, sizeof(double), CompareDouble);
  const double margin_p10 = PercentileSorted(m_sorted, idx, 10.0);
  const double margin_p50 = PercentileSorted(m_sorted, idx, 50.0);

  const double parity_py =
      (idx > 0) ? (100.0 * (double)hits_py / (double)idx) : 0.0;

  FILE* fr = fopen("export/results_batch.json", "wb");
  if (!fr) {
    perror("open results");
    free(buf);
    free(margins);
    free(t_us);
    free(m_sorted);
    free(cm);
    return 1;
  }

  fprintf(fr, "{\n");
  fprintf(fr, "  \"forwards\": %d,\n", idx);
  fprintf(fr, "  \"elapsed_us_total\": %.6f,\n", elapsed_us_total);
  fprintf(fr, "  \"macs_per_inf\": %lld,\n", macs_per_inf);
  fprintf(fr, "  \"ns_per_mac\": %.6f,\n", ns_per_mac);
  fprintf(fr, "  \"parity_py_pct\": %.6f,\n", parity_py);
  fprintf(fr, "  \"margin_p10\": %.6f,\n", margin_p10);
  fprintf(fr, "  \"margin_p50\": %.6f,\n", margin_p50);

  if (correct_lab > 0) {
    const double acc_lab = 100.0 * (double)correct_lab / (double)idx;
    const double loss_mean = loss_sum / (double)idx;
    fprintf(fr, "  \"acc_label_pct\": %.6f,\n", acc_lab);
    fprintf(fr, "  \"loss_mean\": %.6f,\n", loss_mean);
    fprintf(fr, "  \"confusion_matrix\": [\n");
    for (int r = 0; r < MLP_N_OUT; ++r) {
      fprintf(fr, "    [");
      for (int c2 = 0; c2 < MLP_N_OUT; ++c2) {
        const unsigned v = cm[r * MLP_N_OUT + c2];
        fprintf(fr, "%u%s", v, (c2 + 1 < MLP_N_OUT) ? ", " : "");
      }
      fprintf(fr, "]%s\n", (r + 1 < MLP_N_OUT) ? "," : "");
    }
    fprintf(fr, "  ],\n");
  }
  fprintf(fr, "  \"timestamp\": %ld\n", (long)time(NULL));
  fprintf(fr, "}\n");
  fclose(fr);

  printf(
      "Totals: forwards=%d  elapsed=%.3f ms  MACs/inf=%lld  ns/MAC=%.3f  "
      "parity_py=%.2f%%",
      idx, elapsed_us_total / 1000.0, macs_per_inf, ns_per_mac, parity_py);
  if (correct_lab > 0) {
    const double acc_lab = 100.0 * (double)correct_lab / (double)idx;
    const double loss_mean = loss_sum / (double)idx;
    printf("  acc=%.2f%%  loss=%.4f", acc_lab, loss_mean);
  }
  printf("\n");

  free(buf);
  free(margins);
  free(t_us);
  free(m_sorted);
  free(cm);
  return 0;
}

static const char* ResolveDefaultJson(void) {
  static const char* kCands[] = {
      "export/test_batch.json",
      "export/test_vector.json",
      "../export/test_batch.json",
      "../export/test_vector.json",
      "../../export/test_batch.json",
      "../../export/test_vector.json",
      NULL,
  };
  for (int i = 0; kCands[i]; ++i) {
    FILE* f = fopen(kCands[i], "rb");
    if (f) { fclose(f); return kCands[i]; }
  }
  return NULL;
}

int main(int argc, char** argv) {
  const char* path = NULL;
  if (argc > 1) {
    path = argv[1];
  } else {
    path = ResolveDefaultJson();
  }
  if (!path) {
    fprintf(stderr,
            "Não foi possível localizar export/test_batch.json "
            "nem export/test_vector.json.\n");
    return 1;
  }

  FILE* fp = fopen(path, "rb");
  if (!fp) {
    perror("fopen");
    return 1;
  }
  const int c = FirstNonWs(fp);
  fclose(fp);

  if (c == '[') return RunBatch(path);
  if (c == '{') return RunSingle(path);

  fprintf(stderr, "JSON inválido: %s\n", path);
  return 1;
}