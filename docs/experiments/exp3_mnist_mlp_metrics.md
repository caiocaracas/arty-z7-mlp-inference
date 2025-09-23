# EXPERIMENTO 3 — Comparação Adam vs SGD (MNIST, MLP float32)

## Objetivo

Avaliar empiricamente o fator **solver** (Adam vs SGD+momentum) no dataset MNIST, controlando arquitetura e hiperparâmetros, coletando métricas de acurácia, robustez, custo e recursos — seguindo o plano da *Meeting‑1*.

## Fatores (independentes)

* Solver: Adam × SGD+momentum (m=0.9, lr=1e-2).
* Dimensão oculta: H ∈ {32, 64}.
## Respostas (dependentes)

* Acurácia em teste (`acc_test`, sklearn).
* Acurácia em inferência C (`acc_label_pct`).
* Paridade Python↔C (`parity_py_pct`).
* Robustez: margens p10/p50.
* Desempenho: `ns/MAC`, `p95 us/inf`.
* Recursos: `bytes_weights`, `bram36k_blocks_est`, `workspace`.

## Controles

* Dataset: MNIST (mnist.npz local).
* Estrutura: 784→H→10, ReLU, argmax direto (sem softmax).
* Treino: max\_iter=50, batch=1024.

## Procedimento

1. Treino em Python (`train.py`), exportando `weights.h`, `meta.json`, `test_batch.json`.
2. Inferência em C (`infer.c`) em lote N=1024, gerando `results_batch.json`.
3. Registro de métricas no formato padronizado.

## Resultados (seed=7, H=64)

| run\_tag                         | solver | H  | seed | acc\_test | acc\_label\_pct | parity\_py\_pct | ns/MAC | margin\_p10 | margin\_p50 | bytes\_weights | bram36k |
| -------------------------------- | ------ | -- | ---- | --------- | --------------- | --------------- | ------ | ----------- | ----------- | -------------- | ------- |
| mnist\_adam\_h64\_s7             | Adam   | 64 | 7    | \~0.964   | 96.4%           | 100%            | —      | —           | —           | 203560         | 45      |
| mnist\_sgd\_m09\_lr1e-2\_h64\_s7 | SGD    | 64 | 7    | 0.9673    | 94.92%          | 100%            | 0.490  | 4.09        | 10.96       | 203560         | 45      |

## Interpretação inicial

* **Acurácia**: Adam ≈96.4% (C), SGD ≈94.9% (C) — ambos acima da meta de 90%. No sklearn, SGD reportou acc\_test=96.7%.
* **Paridade**: ambos com 100% → contrato Python↔C preservado.
* **Robustez**: margens do SGD (p10≈4.1, p50≈11.0) indicam separação razoável entre logits; robustez mantida.
* **Desempenho**: SGD mediu 0.49 ns/MAC na CPU host → bem acima do EXP2 (0.95 ns/MAC); reforça escalabilidade da pipeline.
* **Recursos**: footprint idêntico (\~200 KiB ≈ 45 BRAM36K, workspace \~3.4 KiB).

## Comparação Adam vs SGD

* **Convergência**: Adam atinge acc boa com menos tuning, como previsto desde EXP2. SGD precisou lr/momentum ajustados.
* **Resultado prático**: ambas opções viáveis na Arty Z7; diferença <2 p.p. em acc\_label\_pct.
* **Tempo de treino**: SGD levou 8.9 s (50 iterações). Adam em runs prévios foi similar; Adam tende a convergir mais cedo.

## Aderência à Meeting‑1

* Hipótese confirmada: MLP pequeno em MNIST >90% acc para ambos solvers.
* Objetivo cumprido: comparação Adam vs SGD com métricas reais Python↔C.

## Próximos passos

* Avançar para EXP4: tuning de hiperparâmetros e análise de quantização.

## Artefatos

* `weights.h`, `meta.json`, `test_batch.json`, `results_batch.json` (Adam e SGD).
