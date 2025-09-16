DESENHO EXPERIMENTAL (MÍNIMO VIÁVEL)
1. Fator (independente)
  - Solver do MLP: Adam fixado, por vias de comparação. Reativar a condição sgd+momentum em branch separada.

2. Respostas (dependentes)
  - Acurácia em teste (acc_test): 0,78375 (~78,4%).
  - Paridade Python↔foward manual: fração de amostras em que predict (sklearn) bate com o forward "na unha" (ReLU + duas matmuls, sem softmax). No run: 1,0 (perfeito).

3. Controles
  - Arquitetura: n_in=16, n_hid=32, n_out=5.
  - Normalização: StandardScaler exportado e reproduzido no C.
  - Critérios de treino: max_iter, early_stopping, validation_fraction fixos no script.

4. Métricas auxiliares
  - Tempo de treino (elapsed_s): 0,8489s (referência de custo).
  - Tamanho do modelo (n_params): 709 (para estimar BRAM/DDR depois).

PROCEDIMENTO (PIPELINE REPRODUTÍVEL)
1. Treino em Python (train.py)
  - Gera/carrega dataset (sintético no MVP), separa treino/teste.
  - Padroniza features (mean/scale).
  - Treina MLP (Adam, ReLU, 1 camada), mede acc_test.
  - Calcula forward manual sem softmax e checa paridade com predict.
  - Exporta:
    - export/weights.h: MLP_N_*, MLP_FEAT_{MEAN,SCALE}, MLP_W{1,2}, MLP_B{1,2} (float32, row-major).
    - export/meta.json: solver, acc_test, elapsed_s, parity, dims, n_params, scaler.
    - export/test_vector.json: um x_raw (não escalado) e pred_argmax_logits do Python (argmax dos logits, sem softmax).

2. Inferência em C (infer.c)
  - Lê x_raw do test_vector.json e normaliza com MLP_FEAT_{MEAN,SCALE}.
  - Faz z1 = x·W1 + b1, h = ReLU(z1), z2 = h·W2 + b2.
  - Decide sem softmax: pred = argmax(z2).
  - Imprime Pred(JSON esperado) e Pred(C calculado).

3. Validação cruzada Python↔C
  - Obtido: Pred(JSON esperado) = 2 e Pred(C calculado) = 2 (OK).
  - Logits mostrados têm máximo em índice 2 (consistente com argmax).
  - Isso confirma que weights + scaler + ordem das matrizes estão corretos.

ARTEFATOS
  - test_vector.json: o caso-teste com x_raw e rótulo previsto (classe 2 no seu run). Usá-lo como prova de paridade para qualquer mudança no C.
  - meta.json: relatório do experimento (métricas e dimensões) e para rastrear regressões.
  - weights.h: contrato de interface PS/PL (float32, row-major), pronto para a Arty mais adiante.

INTERPRETAÇÃO DOS RESULTADOS
  - acc_test ≈ 78%: é referência sobre dados sintéticos. Inválido como métrica de qualidade. Pouco robusto.
  - parity = 1,0: sanidade perfeita; se mexer no C (ex.: otimizações, NEON, quantização), essa métrica deve continuar 1,0 para os vetores-teste.

ADERÊNCIA À MEETING 1
Base mínima local (Python→C), sem softmax; solver comparável (Adam ativo; SGD guardado para experimento posterior); arquivos exportados viabilizam portar para a PS/PL; métricas registradas.

PRÓXIMOS PASSOS
1. Replicações e blocagem: gerar um lote (p.ex. 32 vetores) em test_batch.json e valide em C todos de uma vez (aumenta a confiança). Avaliar acc_test.
2. Fator “solver” (2 níveis: Adam vs SGD+mom): rodar ambos mantendo arquitetura/hiperparâmetros, comaprar acc_test, elapsed_s; registrar em meta.json e em relatório.
3. Fator “dataset”: trocar para MNIST mantendo o mesmo script (caso aprovado em meeting 2); verificar se acc_test sobe para ~90%+.
4. Planejar quantização (novo fator depois): float32 → Q15, medindo impacto na paridade e acc_test.