1. Hipótese: Uma rede neural multicamada pequena (2–3 camadas), treinada em MNIST, atinge acurácia ≥90% usando otimizadores clássicos (sgd com momentum e Adam), mesmo substituindo a função softmax por alternativas menos custosas para a etapa de inferência.

2. Objetivos
	- Primário:  
		    Avaliar a viabilidade de rodar uma rede neural pequena em ambiente híbrido (Python → C), coletando parâmetros reais para futura implementação em FPGA.
	- Secundários:
	    - Comparar desempenho de solvers (sgd + momentum vs Adam).
	    - Medir acurácia e custo computacional em rede simples.
	    - Analisar estrutura interna da rede (nº de pesos, tamanho dos arrays).
	    - Explorar alternativas ao softmax (ex.: argmax direto). -- ReLU
	    
3. Variáveis
	- Independentes:
		- Tipo de solver (sgd com momentum vs adam).
		- Função de ativação de saída (softmax vs aproximações).
	- Dependentes:
		- Acurácia no conjunto de teste (%).
		- Tempo de treinamento (s).
	- Controladas:
		- Dataset (MNIST).
		- Estrutura da rede (mesmo nº de camadas/neurônios).
		- Número de épocas (max_iter).
		- Taxa de aprendizado (learning_rate_init).
	
4. Metodologia
	- Treinamento inicial (Python / scikit-learn)
	    - Dataset: MNIST (784 entradas, 10 classes).
	    - Modelo: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20).
	    - Testar solver=sgd, momentum=0.9 e solver=adam.
	    - Avaliar acurácia e tempo.
	- Exportação dos pesos
	    - Usar clf.coefs e clf.intercepts.
	    - Salvar em formato .npz ou .csv.
	- Inferência manual (Python)
	    - Implementar forward pass “na unha” em Python, usando os pesos exportados.
	    - Comparar com `predict()` do scikit-learn.
	- Implementação em C
	    - Reescrever forward pass em C.
	    - Inicialmente com pesos embutidos (arrays fixos).
	    - Depois, evoluir para leitura de arquivo exportado.
	- Substituição do softmax
	    - Testar saída com argmax direto.
	    - Testar normalização linear simples.
	    - Comparar impacto na acurácia.
	
5. Métricas de avaliação
	- Acurácia (%) no conjunto de teste.
	- Tempo de treino (s).
	- Tamanho dos pesos exportados (KB/MB).
	- Consistência da inferência (Python vs C).	

 6. Resultados esperados
	- Adam deve convergir mais rápido que sgd, mantendo acurácia próxima ou superior.
	- Softmax aproximado (argmax ou normalização linear) deve manter acurácia aceitável (>90%).
	- Quantidade de parâmetros permitirá estimar uso de BRAM/DDR na Arty Z7.
	- Código C validará portabilidade e modularização para implementação futura em RTL.
