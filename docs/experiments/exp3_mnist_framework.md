# EXPERIMENTO 3 — MNIST (MLP float32) • Avaliação de Métricas e Escalabilidade do Framework

## 1) Objetivo
Consolidar o framework de **exportação Python → inferência C → Arty Z7** em um dataset real (MNIST), comparando ambos solvers Adam e SGD, avaliando métricas de acurácia, robustez, custo e recursos, e verificando a **escalabilidade do pipeline** para futura aplicação na Arty Z7. 

---

## 2) Fatores (independentes)
- **Solver**: {Adam, SGD+momentum (m=0.9, lr=1e-2)}  
- **Dimensão oculta**: H ∈ {32, 64}  

---

## 3) Respostas (dependentes)
- **Qualidade**:  
  - `acc_test` (Python/sklearn, referência)  
  - `acc_label_pct` (C)  
- **Contrato**: `parity_py_pct` (Python↔C)  
- **Robustez**: margens p10, p50  
- **Desempenho**: `ns_per_mac`, `p95_us_per_inf`  
- **Recursos**: `bytes_weights`, `bram36k_blocks_est`, `bytes_workspace`  

---

## 4) Controles
- **Dataset**: MNIST (`mnist.npz`, local).  
- **Arquitetura**: 784→H→10, ativação ReLU, saída por `argmax(logits)` (sem softmax).  
- **Treino (andaime sklearn)**: `max_iter=50`, `batch=1024`.  
- **Exportação**: `weights.h`, `meta.json`, `test_batch.json` (Python).  
- **Inferência (produto)**: `infer.c`, recebendo `test_batch.json` e gerando `results_batch.json`.  

---

## 5) Procedimento
1. **Treino (Python/sklearn)**  
   - Executado com Adam e SGD+momentum.  
   - Export de pesos + normalização para `weights.h`.  
   - Artefatos: `meta.json`, `test_batch.json`.  

2. **Inferência (C)**  
   - Rodado sobre lote N=1024 (`test_batch.json`).  
   - Gerado `results_batch.json` com métricas de acurácia, paridade, latência, margens e footprint.  

3. **Verificação de contrato**  
   - Confirmado `parity_py_pct = 100%`.  

---

## 6) Plano de amostragem
- Lote de teste fixo de **N=1024** amostras do MNIST (estratificado).  
- Seed=7 usada para reprodutibilidade (mantida em Adam e SGD).  
- Topologia H=64 avaliada.

---

## 7) Análise planejada
- **MACs/inf**: calcular `784×H + H×10`. Para H=64 → 50,816 MACs.  
- **ns/MAC**: derivado de `elapsed_us_total / (forwards × MACs/inf)`.  
- **Recursos**: pesos ×4 bytes (float32), dividido por 36 KiB → BRAM36K estimada.
- **Robustez**: avaliar margens p10 e p50 no lote.  
- **Foco**: validar escalabilidade do framework (contrato de export/import, métricas consistentes e footprint plausível para a Arty Z7).  
- **Comparação de solver**: análise secundária, apenas para mostrar consistência do framework.  

---

## 8) Resultados (seed=7, H=64)

| run_tag                         | solver | H  | seed | acc_test (Py) | acc_label_pct (C) | parity_py_pct | ns/MAC | margin_p10 | margin_p50 | bytes_weights | BRAM36K | workspace |
|---------------------------------|--------|----|------|---------------|-------------------|---------------|--------|------------|------------|---------------|---------|-----------|
| mnist_adam_h64_s7               | Adam   | 64 | 7    | ~0.964        | 96.4%             | 100%          | ~0.49  | —          | —          | 203,560       | ~45     | ~3.4 KiB  |
| mnist_sgd_m09_lr1e-2_h64_s7     | SGD    | 64 | 7    | 0.9673        | 94.9%             | 100%          | 0.490  | 4.09       | 10.96      | 203,560       | ~45     | ~3.4 KiB  |

---

## 9) Interpretação
- **Framework**: export/import estável, paridade plena, métricas consistentes → contrato validado.  
- **Qualidade**: ambos >90% em C, confirmando viabilidade.  
- **Desempenho**: ~0.49 ns/MAC (host), metade do EXP2 (~0.95 ns/MAC), mostrando escalabilidade.  
- **Robustez**: margens elevadas suportam quantização/approx.  
- **Recursos**: footprint (~200 KiB + 3.4 KiB workspace) cabe em BRAM (~45 blocos) ou pode ser tileado via DDR.  

---

## 10) Comparação secundária Adam vs SGD
- Adam convergiu sem tuning, SGD exigiu lr/momentum.  
- Acurácia próxima; diferença <2 p.p. em `acc_label_pct`.  
- Ambos viáveis como andaime de treino, sem impacto na inferência.  

---

## 11) Aderência ao Plano (Meeting-1)
- Objetivo central atingido: **framework com métricas escaláveis e paridade validada**, pronto para portar à Arty Z7.  
- Análise secundária de solver demonstra robustez do pipeline.  
- EXP3 fecha a etapa de baseline real (MNIST).  
- Prepara terreno para **EXP4: tuning/quantização**.  

---

## 12) Ameaças à validade
- Latência medida em CPU host, não na PS/PL.  
- Apenas um run completo; replicações opcionais.  

---

## 13) Conclusão
O **foco do EXP3** foi alcançado: validar o framework de métricas e escalabilidade no MNIST, consolidando um pipeline pronto para a Arty Z7.  
- Export/import estáveis, paridade confirmada.  
- Acurácia ≥95% possível.  
- Footprint compatível com a placa.  
- Latência por MAC em patamar competitivo.  

Próxima etapa (**EXP4**) é **explorar tuning e quantização**, usando margens como métrica de robustez.  

---

## 14) Artefatos
- `weights.h`, `meta.json`, `test_batch.json` (Python)  
- `results_batch.json` (C)  
