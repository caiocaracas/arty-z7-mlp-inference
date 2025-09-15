import json, pathlib, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUTDIR = pathlib.Path("export")
OUTDIR.mkdir(exist_ok=True)
# place-holder
hidden_units = 32
adam_cfg = dict(solver="adam", learning_rate_init=1e-3, beta_1=0.9, beta_2=0.999)
sgd_cfg  = dict(solver="sgd",  learning_rate_init=5e-3, momentum=0.9, nesterovs_momentum=True)

X, y = make_classification(
    n_samples=4000, n_features=16, n_informative=12, n_redundant=0,
    n_classes=5, random_state=42
)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# MLP => scale sensitive
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xte_s = scaler.transform(Xte)

def train_eval(cfg, max_iter=200, rs=0):
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_units,), activation="relu",
        alpha=1e-4, max_iter=max_iter, early_stopping=True,
        n_iter_no_change=10, random_state=rs, **cfg
    )
    clf.fit(Xtr_s, ytr)
    y_pred = clf.predict(Xte_s)
    acc = accuracy_score(yte, y_pred)
    return clf, acc
# adam x sgd (m)
clf_adam, acc_adam = train_eval(adam_cfg)
clf_sgd,  acc_sgd  = train_eval(sgd_cfg)

if acc_adam >= acc_sgd:
  best_name, best_acc, clf = "adam", acc_adam, clf_adam
else: 
  best_name, best_acc, clf = "sgd_momentum", acc_sgd, clf_sgd
print(f"[VAL] Adam={acc_adam:.4f}  SGD(m)={acc_sgd:.4f}  -> best={best_name}")

n_in  = clf.coefs_[0].shape[0]
n_hid = clf.coefs_[0].shape[1]
n_out = clf.coefs_[1].shape[1]
#row-major => C-friendly
W1 = clf.coefs_[0].astype(np.float32)      
b1 = clf.intercepts_[0].astype(np.float32) 
W2 = clf.coefs_[1].astype(np.float32)       
b2 = clf.intercepts_[1].astype(np.float32)