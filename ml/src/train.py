#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Treino MLP + export de artefatos e métricas estáveis (Synthetic/MNIST).

Este script treina uma MLP simples (float32 + ReLU) em dados sintéticos ou MNIST,
exporta pesos/escala para um header C, gera metadados e um conjunto de teste
(único vetor e/ou lote) para validação do binário de inferência em C.

Artefatos gerados em `--outdir`:
- weights.h: pesos e parâmetros de normalização (float32).
- meta.json: dimensões, recursos (n_params/bytes/BRAM), macs_per_inf,
  métricas (acc/log-loss/confusion matrix/curva de loss) e paridade Py.
- test_vector.json: 1 amostra (para smoke test).
- test_batch.json: lote com {x_raw, label, pred_py, margin} (se habilitado).

Exemplos:
  # MNIST via OpenML (requer internet no 1º uso; cache local depois)
  python3 train.py --dataset mnist_openml --hidden 64 --max_iter 50 --batch_size 1024

  # MNIST offline (arquivo mnist.npz, mesmo formato do Keras)
  python3 train.py --dataset mnist_npz --mnist_npz ./mnist.npz --hidden 64 --max_iter 50

  # Comparar Adam vs SGD+momentum
  python3 train.py --dataset mnist_openml --solver sgd --momentum 0.9 --lr 1e-2 --tag sgd_m09
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuração
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Parâmetros do experimento e de export."""

    # I/O e identificação
    outdir: Path
    dataset_name: str
    run_tag: str

    # Dados (synthetic) - ignorados para MNIST
    n_samples: int
    n_features: int
    n_classes: int
    class_sep: float

    # Controle experimental
    random_state: int
    test_size: float

    # Modelo/treino
    hidden: int
    max_iter: int
    alpha: float
    batch_size: int

    # Dataset/otimizador
    dataset: str                # "synthetic" | "mnist_openml" | "mnist_npz"
    mnist_npz: Optional[Path]   # caminho para mnist.npz (se dataset==mnist_npz)
    solver: str                 # "adam" | "sgd"
    momentum: float             # usado quando solver==sgd
    lr: float                   # learning_rate_init


DEFAULT_CFG = Config(
    outdir=Path("export"),
    dataset_name="synthetic_clf",
    run_tag="baseline",
    n_samples=6000,
    n_features=16,
    n_classes=4,
    class_sep=1.5,
    random_state=7,
    test_size=0.25,
    hidden=32,
    max_iter=200,
    alpha=1e-4,
    batch_size=1024,
    dataset="synthetic",
    mnist_npz=None,
    solver="adam",
    momentum=0.9,
    lr=1e-3,
)


# =============================================================================
# Rede e utilidades
# =============================================================================


def forward_logits(
    x_s: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """(x_s @ W1 + b1) -> ReLU -> (@ W2 + b2) -> logits (N,C)."""
    h = x_s @ w1 + b1
    np.maximum(h, 0.0, out=h)  # ReLU in-place
    z2 = h @ w2 + b2
    return z2


def logits_from_raw_batch(
    x_raw: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Normaliza um lote cru e computa logits."""
    x_s = (x_raw - mean) / (scale + 1e-12)
    return forward_logits(x_s, w1, b1, w2, b2)


# =============================================================================
# Export: pesos + meta + lotes
# =============================================================================


def _dump_array_c(fh, name: str, arr: np.ndarray) -> None:
    """Escreve um array float32 como literal C, 8 valores por linha."""
    flat = arr.ravel().astype(np.float32, copy=False)
    fh.write(f"static const float {name}[{flat.size}] = {{\n")
    for i in range(0, flat.size, 8):
        chunk = ", ".join(f"{v:.8e}f" for v in flat[i : i + 8])
        fh.write(f"  {chunk},\n")
    fh.write("};\n\n")


def export_header_and_meta(
    cfg: Config,
    scaler: StandardScaler,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    solver_name: str,
    acc_test: float,
    elapsed_s: float,
    parity_py: float,
    extra_metrics: Dict | None = None,
) -> None:
    """Grava `weights.h` e `meta.json` no diretório de saída."""
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    n_in, n_hid = w1.shape
    _, n_out = w2.shape
    n_params = int(w1.size + b1.size + w2.size + b2.size)

    bytes_per_weight = 4
    bytes_weights = int(n_params * bytes_per_weight)
    bram36k_bytes = 4608  # aproximação conservadora
    bram36k_blocks_est = int((bytes_weights + bram36k_bytes - 1) // bram36k_bytes)
    macs_per_inf = int(n_in * n_hid + n_hid * n_out)

    header = cfg.outdir / "weights.h"
    with header.open("w", encoding="utf-8") as fh:
        fh.write("// Gerado por train.py — MLP float32, ReLU.\n")
        fh.write("#pragma once\n#include <stddef.h>\n\n")
        fh.write(f"#define MLP_N_IN  {n_in}\n")
        fh.write(f"#define MLP_N_HID {n_hid}\n")
        fh.write(f"#define MLP_N_OUT {n_out}\n\n")

        mean = scaler.mean_.astype(np.float32, copy=False).ravel()
        scale = scaler.scale_.astype(np.float32, copy=False).ravel()

        _dump_array_c(fh, "MLP_FEAT_MEAN", mean)
        _dump_array_c(fh, "MLP_FEAT_SCALE", scale)

        # Pesos em row-major por neurônio de saída.
        # Para W1 (D,H), gravar como (H,D) row-major: W1.T.flatten().
        _dump_array_c(fh, "MLP_W1", w1.T.flatten())
        _dump_array_c(fh, "MLP_B1", b1)
        # Para W2 (H,C), gravar como (C,H) row-major: W2.T.flatten().
        _dump_array_c(fh, "MLP_W2", w2.T.flatten())
        _dump_array_c(fh, "MLP_B2", b2)

    meta = {
        "dataset": cfg.dataset_name,
        "run_tag": cfg.run_tag,
        "dims": {"n_in": n_in, "n_hid": n_hid, "n_out": n_out},
        "solver": solver_name,
        "dtype_weights": "float32",
        "metrics_test": {"acc": float(acc_test)},
        "train_elapsed_s": float(elapsed_s),
        "macs_per_inf": macs_per_inf,
        "resources": {
            "n_params": n_params,
            "bytes_per_weight": bytes_per_weight,
            "bytes_weights": bytes_weights,
            "bram36k_blocks_est": bram36k_blocks_est,
        },
        "parity_py_pct": float(parity_py * 100.0),
        "scaler": {
            "mean": mean.astype(np.float32).tolist(),
            "scale": scale.astype(np.float32).tolist(),
        },
        "timestamp": int(time.time()),
    }
    if extra_metrics:
        # Campos em extra sobrescrevem os padrões quando coincidirem.
        meta.update(extra_metrics)

    (cfg.outdir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    logging.info("weights.h e meta.json gravados em: %s", cfg.outdir)


def save_test_vector(
    outdir: Path,
    x_raw: np.ndarray,
    scaler: StandardScaler,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> None:
    """Grava um único vetor de teste com pred e margem."""
    z2 = logits_from_raw_batch(
        x_raw[None, :], scaler.mean_, scaler.scale_, w1, b1, w2, b2
    )[0]
    pred = int(np.argmax(z2))
    top2 = np.partition(z2, -2)[-2:]
    top2.sort()
    margin = float(top2[1] - top2[0])

    payload = {
        "x_raw": x_raw.astype(np.float32).tolist(),
        "pred_py": pred,
        "margin": margin,
    }
    (outdir / "test_vector.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logging.info("test_vector.json (1 amostra) em: %s", outdir)


def save_test_batch(
    outdir: Path,
    xte_raw: np.ndarray,
    yte: np.ndarray,
    scaler: StandardScaler,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    batch_size: int,
) -> None:
    """Grava lote de teste com x_raw, label, pred_py e margem."""
    n = min(batch_size, xte_raw.shape[0])
    xsel = np.asarray(xte_raw[:n], dtype=np.float32)
    ysel = np.asarray(yte[:n], dtype=np.int32)

    z2 = logits_from_raw_batch(xsel, scaler.mean_, scaler.scale_, w1, b1, w2, b2)
    pred = np.argmax(z2, axis=1).astype(np.int32)
    top2 = np.partition(z2, kth=-2, axis=1)[:, -2:]
    top2.sort(axis=1)
    margins = (top2[:, 1] - top2[:, 0]).astype(np.float32)

    batch: List[Dict] = []
    for i in range(n):
        batch.append(
            {
                "x_raw": xsel[i].tolist(),
                "label": int(ysel[i]),
                "pred_py": int(pred[i]),
                "margin": float(margins[i]),
            }
        )

    (outdir / "test_batch.json").write_text(
        json.dumps(batch, indent=2), encoding="utf-8"
    )
    logging.info("test_batch.json (N=%d) em: %s", n, outdir)


# =============================================================================
# Dados
# =============================================================================


def load_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, Config]:
    """Carrega X, y de acordo com cfg.dataset; pode ajustar campos do cfg."""
    if cfg.dataset == "synthetic":
        from sklearn.datasets import make_classification

        x, y = make_classification(
            n_samples=cfg.n_samples,
            n_features=cfg.n_features,
            n_informative=cfg.n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=cfg.n_classes,
            class_sep=cfg.class_sep,
            random_state=cfg.random_state,
        )
        # dataset_name já é "synthetic_clf"
        return x.astype(np.float32), y.astype(np.int32), cfg

    if cfg.dataset == "mnist_openml":
        # 1º uso baixa; sklearn guarda cache em ~/.sklearn/openml
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
        x = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        # Fixar dimensões no cfg retornado
        new_cfg = replace_cfg(cfg, dataset_name="mnist_openml", n_features=784, n_classes=10)
        return x, y, new_cfg

    if cfg.dataset == "mnist_npz":
        if cfg.mnist_npz is None:
            raise ValueError("Forneça --mnist_npz /caminho/para/mnist.npz")
        data = np.load(cfg.mnist_npz)
        x_tr = data["x_train"].reshape(-1, 28 * 28).astype(np.float32) / 255.0
        y_tr = data["y_train"].astype(np.int32)
        x_te = data["x_test"].reshape(-1, 28 * 28).astype(np.float32) / 255.0
        y_te = data["y_test"].astype(np.int32)
        x = np.concatenate([x_tr, x_te], axis=0)
        y = np.concatenate([y_tr, y_te], axis=0)
        new_cfg = replace_cfg(cfg, dataset_name="mnist_npz", n_features=784, n_classes=10)
        return x, y, new_cfg

    raise ValueError(f"Dataset desconhecido: {cfg.dataset}")


def replace_cfg(cfg: Config, **kwargs) -> Config:
    """Cria uma cópia imutável de cfg alterando somente os campos desejados."""
    d = cfg.__dict__.copy()
    d.update(kwargs)
    return Config(**d)


# =============================================================================
# Execução
# =============================================================================


def build_config_from_flags() -> Config:
    """Cria a configuração a partir de flags de linha de comando."""
    p = argparse.ArgumentParser(
        description="Treino MLP e export de artefatos/métricas."
    )
    # I/O e identificação
    p.add_argument("--outdir", type=Path, default=DEFAULT_CFG.outdir, help="Diretório de saída.")
    p.add_argument("--tag", type=str, default=DEFAULT_CFG.run_tag, help="Identificador de execução (run_tag).")

    # Controle experimental
    p.add_argument("--seed", type=int, default=DEFAULT_CFG.random_state, help="Semente de aleatoriedade.")
    p.add_argument("--batch_size", type=int, default=DEFAULT_CFG.batch_size, help="Tamanho do lote de teste (0 desabilita).")
    p.add_argument("--max_iter", type=int, default=DEFAULT_CFG.max_iter, help="Iterações máximas do MLPClassifier.")
    p.add_argument("--hidden", type=int, default=DEFAULT_CFG.hidden, help="Número de neurônios na camada escondida.")

    # Dados (synthetic)
    p.add_argument("--samples", type=int, default=DEFAULT_CFG.n_samples, help="Número de amostras sintéticas.")
    p.add_argument("--features", type=int, default=DEFAULT_CFG.n_features, help="Número de features (synthetic).")
    p.add_argument("--classes", type=int, default=DEFAULT_CFG.n_classes, help="Número de classes (synthetic).")
    p.add_argument("--class_sep", type=float, default=DEFAULT_CFG.class_sep, help="Separabilidade (synthetic).")

    # Dataset/otimizador
    p.add_argument("--dataset", type=str, choices=["synthetic", "mnist_openml", "mnist_npz"], default=DEFAULT_CFG.dataset)
    p.add_argument("--mnist_npz", type=Path, default=None, help="Caminho para mnist.npz (quando --dataset=mnist_npz).")

    p.add_argument("--solver", type=str, choices=["adam", "sgd"], default=DEFAULT_CFG.solver)
    p.add_argument("--momentum", type=float, default=DEFAULT_CFG.momentum, help="Momentum (apenas para --solver=sgd).")
    p.add_argument("--lr", type=float, default=DEFAULT_CFG.lr, help="learning_rate_init do MLPClassifier.")

    args = p.parse_args()

    return Config(
        outdir=args.outdir,
        dataset_name=DEFAULT_CFG.dataset_name,
        run_tag=args.tag,
        n_samples=args.samples,
        n_features=args.features,
        n_classes=args.classes,
        class_sep=args.class_sep,
        random_state=args.seed,
        test_size=DEFAULT_CFG.test_size,
        hidden=args.hidden,
        max_iter=args.max_iter,
        alpha=DEFAULT_CFG.alpha,
        batch_size=args.batch_size,
        dataset=args.dataset,
        mnist_npz=args.mnist_npz,
        solver=args.solver,
        momentum=args.momentum,
        lr=args.lr,
    )


def main() -> None:
    """Fluxo principal: dados → treino → métricas → export."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    cfg = build_config_from_flags()

    # Carrega dataset conforme cfg.dataset
    x, y, cfg = load_dataset(cfg)

    # Split estratificado / controle de variância (DAE)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # Normalização (mantém contrato com inferência em C)
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)

    # Modelo / Solver
    if cfg.solver == "sgd":
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.hidden,),
            activation="relu",
            solver="sgd",
            momentum=cfg.momentum,
            learning_rate_init=cfg.lr,
            alpha=cfg.alpha,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            verbose=False,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg.hidden,),
            activation="relu",
            solver="adam",
            learning_rate_init=cfg.lr,
            alpha=cfg.alpha,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            verbose=False,
        )

    # Treino
    t0 = time.perf_counter()
    clf.fit(x_tr_s, y_tr)
    train_elapsed = time.perf_counter() - t0

    # Métricas de teste
    y_pred = clf.predict(x_te_s)
    acc = accuracy_score(y_te, y_pred)
    try:
        y_proba = clf.predict_proba(x_te_s)
        test_ll = float(log_loss(y_te, y_proba, labels=list(range(int(np.max(y) + 1)))))
    except Exception:
        test_ll = float("nan")

    cm = confusion_matrix(y_te, y_pred).astype(int).tolist()
    class_balance_test = np.bincount(y_te, minlength=int(np.max(y) + 1)).astype(int).tolist()
    train_loss_final = float(getattr(clf, "loss_", float("nan")))
    train_loss_curve = [float(v) for v in getattr(clf, "loss_curve_", [])]

    # Pesos
    w1 = np.ascontiguousarray(clf.coefs_[0].astype(np.float32, copy=False))
    b1 = np.ascontiguousarray(clf.intercepts_[0].astype(np.float32, copy=False))
    w2 = np.ascontiguousarray(clf.coefs_[1].astype(np.float32, copy=False))
    b2 = np.ascontiguousarray(clf.intercepts_[1].astype(np.float32, copy=False))

    # Paridade (forward Py manual vs sklearn)
    z2_chk = logits_from_raw_batch(
        x_te[:min(256, x_te.shape[0])], scaler.mean_, scaler.scale_, w1, b1, w2, b2
    )
    y_pred_man = np.argmax(z2_chk, axis=1)
    parity_py = float(np.mean(y_pred_man == clf.predict(x_te_s[:min(256, x_te_s.shape[0])])))

    # Recursos / custos
    n_in, n_hid = w1.shape
    n_out = w2.shape[1]
    macs_per_inf = int(n_in * n_hid + n_hid * n_out)

    bytes_per_weight = 4  # float32
    n_params = int(w1.size + b1.size + w2.size + b2.size)
    bytes_weights = int(n_params * bytes_per_weight)
    bram36k_bytes = 4608
    bram36k_blocks_est = int((bytes_weights + bram36k_bytes - 1) // bram36k_bytes)

    # Buffers de execução (float32)
    bytes_x = n_in * 4
    bytes_hidden = n_hid * 4
    bytes_out = n_out * 4
    workspace_bytes = bytes_x + bytes_hidden + bytes_out
    activations_peak_bytes = max(bytes_x, bytes_hidden, bytes_out)

    def kib(xi: int) -> float:
        return round(xi / 1024.0, 3)

    def mib(xi: int) -> float:
        return round(xi / (1024.0 ** 2), 3)

    extra = {
        "test_log_loss": test_ll,
        "confusion_matrix": cm,
        "class_balance_test": class_balance_test,
        "train_loss_final": train_loss_final,
        "train_loss_curve": train_loss_curve,

        # custo computacional
        "macs_per_inf": macs_per_inf,

        # pesos (modelo)
        "resources": {
            "n_params": n_params,
            "dtype_weights": "float32",
            "bytes_per_weight": bytes_per_weight,
            "bytes_weights": bytes_weights,
            "bytes_weights_kib": kib(bytes_weights),
            "bytes_weights_mib": mib(bytes_weights),
            "bram36k_blocks_est": bram36k_blocks_est,
        },

        # buffers de execução (referência para dimensionar BRAM/DDR na Arty Z7)
        "runtime_buffers": {
            "dtype_activations": "float32",
            "bytes_x": bytes_x,
            "bytes_hidden": bytes_hidden,
            "bytes_out": bytes_out,
            "workspace_bytes": workspace_bytes,
            "workspace_kib": kib(workspace_bytes),
            "activations_peak_bytes": activations_peak_bytes,
        },

        # energia: placeholder para próxima fase
        "energy_per_mac_pJ": None,
    }

    # Export
    export_header_and_meta(
        cfg=cfg,
        scaler=scaler,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        solver_name=cfg.solver,
        acc_test=acc,
        elapsed_s=train_elapsed,
        parity_py=parity_py,
        extra_metrics=extra,
    )

    # Conjuntos de teste
    save_test_vector(cfg.outdir, x_te[0], scaler, w1, b1, w2, b2)
    if cfg.batch_size > 0:
        save_test_batch(cfg.outdir, x_te, y_te, scaler, w1, b1, w2, b2, cfg.batch_size)


if __name__ == "__main__":
    main()
