import json, pathlib, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTDIR = pathlib.Path("export")
OUTDIR.mkdir(exist_ok=True)
hidden_units = 32
adam_cfg = dict(solver="adam", learning_rate_init=1e-3, beta_1=0.9, beta_2=0.999)
sgd_cfg  = dict(solver="sgd", learning_rate_init=5e-3, momentum=0.9, nesterovs_momentum=True)

X, y = make_classification(n_samples=4000, n_features=16, n_informative=12,
                           n_redundant=0, n_classes=5, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

