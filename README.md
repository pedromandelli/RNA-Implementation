# Implementação de Rede Neural Artificial

Este projeto implementa uma Rede Neural Artificial (RNA) em Python, utilizando os conceitos teóricos de redes neurais e recursos de baixo nível da linguagem, sem depender de bibliotecas de alto nível para a implementação da rede.

## Estrutura do Projeto

```
.
├── src/
│   └── rna/            # Pacote principal da rede neural
│       ├── __init__.py # Inicializador do pacote
│       └── network.py  # Implementação da classe NeuralNetwork
├── tests/              # Testes unitários
└── notebooks/          # Notebooks Jupyter para demonstração
```

## Funcionalidades

A implementação inclui:

- Estrutura da rede neural (camadas, pesos, biases)
- Três funções de ativação: ReLU, Sigmoid e Softmax
- Duas funções de perda: MSE e Cross-Entropy
- Algoritmo de retropropagação (backpropagation)
- Otimização por gradiente descendente

## Dependências

O projeto utiliza apenas bibliotecas para operações de baixo nível:

- NumPy: para operações matriciais e cálculos numéricos
- Pandas: para manipulação de dados
- Matplotlib: para visualização

## Uso Básico

```python
from src.rna.network import NeuralNetwork

# Criar uma rede com 3 entradas, 2 camadas ocultas (4 e 2 neurônios) e 1 saída
model = NeuralNetwork([3, 4, 2, 1])

# Treinar o modelo
model.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

# Fazer predições
predictions = model.predict(X_test)
```

## Equipe

*A ser preenchido*

## Licença

MIT 