#train.py
from __future__ import annotations
import json
import time
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

def ensure_outdir(p: Path) -> None:
  p.mkdir(parents=True, exist_ok=True)

def write_c_array(f, name: str, arr: np.ndarray) -> None:
  """escreve um array C (float32) em linhas compactas (8 valores/linha)."""
  flat = np.asarray(arr, dtype=np.float32).ravel()
  f.write(f"static const int {name}_LEN = {flat.size};\n")
  f.write(f"static const float {name}[] = {{\n")
  for i in range(0, flat.size, 8):
    chunk = ", ".join(f"{v:.8e}f" for v in flat[i:i+8])
    f.write(f"  {chunk},\n")
  f.write("};\n\n")

def export_header_and_meta(
  outdir: Path,
  scaler: StandardScaler,
  W1: np.ndarray, b1: np.ndarray,
  W2: np.ndarray, b2: np.ndarray,
  solver_name: str,
  acc_test: float
) -> None:
  header = outdir / "weights.h"
  with header.open("w", encoding="utf-8") as f:
    n_in, n_hid = W1.shape
    _, n_out = W2.shape
    f.write("// weights.h - gerado por train_mlp_adam_only.py\n")
    f.write("// Formato row-major:\n")
    f.write("//   W1: (n_in x n_hid), b1: (n_hid)\n")
    f.write("//   W2: (n_hid x n_out), b2: (n_out)\n\n")
    f.write("#pragma once\n#include <stdint.h>\n\n")
    f.write(f"static const int MLP_N_IN  = {n_in};\n")
    f.write(f"static const int MLP_N_HID = {n_hid};\n")
    f.write(f"static const int MLP_N_OUT = {n_out};\n\n")

    # scaler para normalização idêntica no C (x - mean)/scale
    write_c_array(f, "MLP_FEAT_MEAN", np.asarray(scaler.mean_, dtype=np.float32))
    write_c_array(f, "MLP_FEAT_SCALE", np.asarray(scaler.scale_, dtype=np.float32))

    # pesos/vieses
    write_c_array(f, "MLP_W1", W1.astype(np.float32, copy=False))
    write_c_array(f, "MLP_B1", b1.astype(np.float32, copy=False))
    write_c_array(f, "MLP_W2", W2.astype(np.float32, copy=False))
    write_c_array(f, "MLP_B2", b2.astype(np.float32, copy=False))

  meta = {
    "solver": solver_name,
    "acc_test": float(acc_test),
    "n_in": int(W1.shape[0]),
    "n_hid": int(W1.shape[1]),
    "n_out": int(W2.shape[1]),
    "feature_mean": np.asarray(scaler.mean_, dtype=np.float32).tolist(),
    "feature_scale": np.asarray(scaler.scale_, dtype=np.float32).tolist()
    }
  (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def save_test_vector(outdir, Xte_raw, scaler, clf):
    x0_raw = np.asarray(Xte_raw[0], dtype=np.float32)
    x0_s = (x0_raw - scaler.mean_) / scaler.scale_
    logits = clf.decision_function([x0_s])[0]
    pred = int(np.argmax(logits))
    payload = {"x_raw": x0_raw.tolist(), "pred_argmax_logits": pred}
    Path(outdir, "test_vector.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"[OK] Teste salvo em {outdir}/test_vector.json (pred={pred})")


def main() -> None:
  t0 = time.perf_counter()
  ensure_outdir(CFG.outdir)

  X, y = make_classification(
      n_samples=CFG.n_samples,
      n_features=CFG.n_features,
      n_informative=CFG.n_informative,
      n_redundant=0,
      n_classes=CFG.n_classes,
      random_state=CFG.random_state
  )
  Xtr, Xte, ytr, yte = train_test_split(
      X, y, test_size=CFG.test_size, stratify=y, random_state=CFG.random_state
  )

  scaler = StandardScaler().fit(Xtr)
  Xtr_s = scaler.transform(Xtr)
  Xte_s = scaler.transform(Xte)

  clf = MLPClassifier(
      hidden_layer_sizes=(CFG.hidden_units,),
      activation="relu",
      alpha=CFG.alpha_l2,
      solver="adam",
      learning_rate_init=CFG.lr_adam,
      beta_1=CFG.beta1,
      beta_2=CFG.beta2,
      max_iter=CFG.max_iter,
      early_stopping=CFG.early_stopping,
      n_iter_no_change=CFG.n_iter_no_change,
      random_state=CFG.random_state,
      shuffle=True,
      verbose=False
  )
  clf.fit(Xtr_s, ytr)

  y_pred = clf.predict(Xte_s)
  acc = accuracy_score(yte, y_pred)

  # C-friendly
  W1 = clf.coefs_[0].astype(np.float32, copy=False)      # (n_in, n_hid)
  b1 = clf.intercepts_[0].astype(np.float32, copy=False) # (n_hid,)
  W2 = clf.coefs_[1].astype(np.float32, copy=False)      # (n_hid, n_out)
  b2 = clf.intercepts_[1].astype(np.float32, copy=False) # (n_out,)

  export_header_and_meta(CFG.outdir, scaler, W1, b1, W2, b2, "adam", acc)
  save_test_vector(CFG.outdir, Xte, scaler, clf)

  dt = time.perf_counter() - t0
  print(f"[OK] Solver=adam  acc_test={acc:.4f}  elapsed={dt:.2f}s")
  print(f"[OK] Gerado em: {CFG.outdir}/weights.h, {CFG.outdir}/meta.json, {CFG.outdir}/test_vector.json")


if __name__ == "__main__":
    main()

"""
# ---- SGD + momentum (desativado) ----
clf_sgd = MLPClassifier(
    hidden_layer_sizes=(CFG.hidden_units,),
    activation="relu",
    alpha=CFG.alpha_l2,
    solver="sgd",
    learning_rate_init=5e-3,
    momentum=0.9,
    nesterovs_momentum=True,
    max_iter=CFG.max_iter,
    early_stopping=CFG.early_stopping,
    n_iter_no_change=CFG.n_iter_no_change,
    random_state=CFG.random_state,
    shuffle=True,
    verbose=False
)
clf_sgd.fit(Xtr_s, ytr)
y_pred_sgd = clf_sgd.predict(Xte_s)
acc_sgd = accuracy_score(yte, y_pred_sgd)
print(f"[INFO] SGD(m): acc_test={acc_sgd:.4f}")
"""