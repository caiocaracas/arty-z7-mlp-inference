# ml/src/train.py
from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

@dataclass(frozen=True)
class Config:
  outdir: Path = Path("export")
  hidden_units: int = 32
  alpha_l2: float = 1e-4
  max_iter: int = 300
  early_stopping: bool = True
  validation_fraction: float = 0.10
  n_iter_no_change: int = 12
  random_state: int = 42
  lr_adam: float = 1e-3
  beta1: float = 0.9
  beta2: float = 0.999
  n_samples: int = 4000
  n_features: int = 16
  n_informative: int = 12
  n_classes: int = 5
  test_size: float = 0.2

CFG = Config()
np.random.seed(CFG.random_state)

def ensure_outdir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)

def write_c_array(f, name: str, arr: np.ndarray) -> None:
  flat = np.asarray(arr, dtype=np.float32, order="C").ravel()
  f.write(f"static const int {name}_LEN = {flat.size};\n")
  f.write(f"static const float MLP_ALIGN {name}[] = {{\n")
  for i in range(0, flat.size, 8):
    chunk = ", ".join(f"{v:.8e}f" for v in flat[i:i+8])
    f.write(f"  {chunk},\n")
  f.write("};\n\n")

def export_header_and_meta(outdir: Path, scaler: StandardScaler,
                          W1: np.ndarray, b1: np.ndarray,
                          W2: np.ndarray, b2: np.ndarray,
                          solver_name: str, acc_test: float,
                          elapsed_s: float, parity: float) -> None:
  n_in, n_hid = W1.shape
  _, n_out  = W2.shape
  header = outdir / "weights.h"
  with header.open("w", encoding="utf-8") as f:
    f.write("#pragma once\n#include <stdint.h>\n\n")
    f.write("#if defined(__GNUC__)\n  #define MLP_ALIGN __attribute__((aligned(16)))\n#else\n  #define MLP_ALIGN\n#endif\n\n")
    f.write(f"static const int MLP_N_IN  = {n_in};\n")
    f.write(f"static const int MLP_N_HID = {n_hid};\n")
    f.write(f"static const int MLP_N_OUT = {n_out};\n\n")
    write_c_array(f, "MLP_FEAT_MEAN", np.asarray(scaler.mean_,  dtype=np.float32))
    write_c_array(f, "MLP_FEAT_SCALE", np.asarray(scaler.scale_, dtype=np.float32))
    write_c_array(f, "MLP_W1", W1)
    write_c_array(f, "MLP_B1", b1)
    write_c_array(f, "MLP_W2", W2)
    write_c_array(f, "MLP_B2", b2)

  n_params = int(W1.size + b1.size + W2.size + b2.size)
  meta = {
    "solver": solver_name,
    "acc_test": float(acc_test),
    "elapsed_s": float(elapsed_s),
    "parity_sklearn_forward": float(parity),
    "n_in": int(n_in), "n_hid": int(n_hid), "n_out": int(n_out),
    "n_params": n_params,
    "feature_mean": np.asarray(scaler.mean_,  dtype=np.float32).tolist(),
    "feature_scale": np.asarray(scaler.scale_, dtype=np.float32).tolist()
  }
  (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def relu_inplace(v: np.ndarray) -> None:
  np.maximum(v, 0.0, out=v)

def forward_logits_xraw(x_raw: np.ndarray,
                      scaler_mean: np.ndarray, scaler_scale: np.ndarray,
                      W1: np.ndarray, b1: np.ndarray,
                      W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
  x = (x_raw.astype(np.float32) - scaler_mean.astype(np.float32)) / (scaler_scale.astype(np.float32) + 1e-12)
  z1 = x @ W1 + b1
  relu_inplace(z1)
  z2 = z1 @ W2 + b2
  return z2

def save_test_vector(outdir: Path, Xte_raw: np.ndarray, scaler: StandardScaler,
                    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> None:
  x0_raw = np.asarray(Xte_raw[0], dtype=np.float32)
  logits = forward_logits_xraw(x0_raw, scaler.mean_, scaler.scale_, W1, b1, W2, b2)
  pred = int(np.argmax(logits))
  payload = {"x_raw": x0_raw.tolist(), "pred_argmax_logits": pred}
  (outdir / "test_vector.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main() -> None:
  t0 = time.perf_counter()
  ensure_outdir(CFG.outdir)

  X, y = make_classification(n_samples=CFG.n_samples, n_features=CFG.n_features,
                              n_informative=CFG.n_informative, n_redundant=0,
                              n_classes=CFG.n_classes, random_state=CFG.random_state)
  Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=CFG.test_size,
                                        stratify=y, random_state=CFG.random_state)

  scaler = StandardScaler().fit(Xtr)
  Xtr_s = scaler.transform(Xtr)
  Xte_s = scaler.transform(Xte)

  clf = MLPClassifier(hidden_layer_sizes=(CFG.hidden_units,), activation="relu",
                      alpha=CFG.alpha_l2, solver="adam",
                      learning_rate_init=CFG.lr_adam, beta_1=CFG.beta1, beta_2=CFG.beta2,
                      max_iter=CFG.max_iter, early_stopping=CFG.early_stopping,
                      validation_fraction=CFG.validation_fraction,
                      n_iter_no_change=CFG.n_iter_no_change, random_state=CFG.random_state,
                      shuffle=True, verbose=False)
  clf.fit(Xtr_s, ytr)

  y_pred = clf.predict(Xte_s)
  acc = accuracy_score(yte, y_pred)

  W1 = np.ascontiguousarray(clf.coefs_[0].astype(np.float32, copy=False))
  b1 = clf.intercepts_[0].astype(np.float32, copy=False)
  W2 = np.ascontiguousarray(clf.coefs_[1].astype(np.float32, copy=False))
  b2 = clf.intercepts_[1].astype(np.float32, copy=False)

  Xchk = Xte[:256].astype(np.float32)
  z1 = (Xchk - scaler.mean_) / (scaler.scale_ + 1e-12)
  z1 = z1 @ W1 + b1
  np.maximum(z1, 0.0, out=z1)
  z2 = z1 @ W2 + b2
  y_pred_man = z2.argmax(axis=1)
  parity = float((y_pred_man == clf.predict((Xchk - scaler.mean_) / (scaler.scale_ + 1e-12))).mean())
  dt = time.perf_counter() - t0

  export_header_and_meta(CFG.outdir, scaler, W1, b1, W2, b2,
                          solver_name="adam", acc_test=acc,
                          elapsed_s=dt, parity=parity)
  save_test_vector(CFG.outdir, Xte, scaler, W1, b1, W2, b2)

if __name__ == "__main__":
    main()