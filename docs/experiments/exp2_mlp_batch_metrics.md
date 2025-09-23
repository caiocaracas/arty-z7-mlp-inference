# EXPERIMENTO 2 — Lote & Métricas de Robustez/Desempenho (MLP float32)

## Objetivo
Consolidar o framework de **inferência em lote** com:
- qualidade (acurácia, loss média, matriz de confusão),
- robustez (margem de decisão p10/p50),
- custo computacional (MACs/inf, ns/MAC),
- recursos (memória de pesos, workspace e BRAM36K estimada),
- **paridade Python↔C** em escala.

## Fatores (independentes)
- Arquitetura: MLP 16→32→4, ReLU.  
- Solver: Adam (fixo). Teste comparativo com SGD+momentum foi analisado teoricamente.  
- Dados: dataset sintético.

## Respostas (dependentes)
- **Acurácia** (`acc_label_pct`)  
- **Loss média** (`loss_mean`)  
- **Paridade** (`parity_py_pct`)  
- **Margens** (`margin_p10`, `margin_p50`)  
- **Desempenho**: `ns/MAC`, `forwards`  
- **Recursos**: `bytes_weights`, `bytes_workspace`, `bram36k_blocks_est`

## Controles
- Normalização e layout de pesos fixos (`weights.h`).  
- Lote exportado (`test_batch.json`).

## Procedimento
1. **Treino (Python)**: exporta `weights.h`, `meta.json`, `test_batch.json`.  
2. **Inferência (C)**: roda `infer`, gera `results_batch.json` com métricas.

## Plano de amostragem
- Lote de **N=1024** amostras.

## Análise planejada
- `MACs/inf = 640`;  
- `ns/MAC` derivado de tempo total;  
- Recursos calculados como em EXP1.

## Resultados (run atual)
- **forwards**: 1024  
- **acc_label_pct**: **94.824%**  
- **loss_mean**: **0.1925**  
- **parity_py_pct**: **100.0%**  
- **margens**: p10 = **1.6388**, p50 = **5.6370**  
- **ns/MAC**: **0.946**  
- **recursos**: pesos=2.64 KiB (~1 BRAM36K), workspace=208 B  
- **confusion_matrix**: erros distribuídos, sem viés dominante.

## Interpretação
- **Consistência**: parity=100% → contrato Python↔C sólido.  
- **Qualidade**: acc≈95%, loss≈0.19.  
- **Robustez**: margens positivas.  
- **Recursos**: footprint mínimo, confirmando viabilidade na Arty Z7.

## Comparação de solvers
- **Overhead** de update: Adam ≈ 12 FLOPs/param (8k); SGD+mom ≈ 4 FLOPs/param (2.7k).  
- **Gradiente/backprop** custa ≈ 384k MACs/passo (B=200).  
- Diferença entre solvers no custo é <2%.  
- **Adam converge em menos passos** sem tuning → menor custo total na prática.  
- Conclusão: Adam segue mais eficiente para este setup.

## Ameaças à validade
- Dados sintéticos; performance real precisa de MNIST.  
- `ns/MAC` medido em CPU host, não reflete PS/PL.

## Conclusão
- Framework de métricas consolidado.  
- Solver Adam justificado por convergência mais rápida e overhead baixo.

## Próximos passos
- EXP3: MNIST baseline.  
- EXP4: comparar solvers com tuning.  
- EXP5: quantização.  
- EXP6: energia na Arty Z7.

## Artefatos
- `weights.h`, `meta.json`, `test_batch.json`, `results_batch.json`.