# Implementação de Rede Neural Artificial

Este projeto implementa uma Rede Neural Artificial (RNA) em Python, utilizando apenas bibliotecas de baixo nível como NumPy para operações matemáticas.

## Visão Geral

O objetivo deste projeto é implementar uma RNA seguindo os princípios matemáticos fundamentais de funcionamento de uma rede neural, incluindo sua estrutura e métodos de treinamento, sem utilizar bibliotecas de alto nível de aprendizado de máquina.

## Componentes Principais

- **Estrutura da Rede Neural**: Implementação de uma rede neural multicamadas com número arbitrário de camadas e neurônios
- **Funções de Ativação**: Implementação de 3 funções de ativação (Sigmoid, ReLU, Tanh)
- **Funções de Perda**: Implementação de 2 funções de perda (MSE, Binary Cross-Entropy)
- **Algoritmo de Retropropagação**: Implementação do algoritmo de backpropagation
- **Otimização**: Implementação do algoritmo de gradiente descendente e suas variações

## Estrutura do Projeto

```
src/
  ├── rna/                          # Código principal da RNA
  │   ├── __init__.py               # Exporta classes principais
  │   ├── network.py                # Classe principal da rede neural
  │   ├── activation_functions/     # Implementações de funções de ativação
  │   │   ├── __init__.py
  │   │   ├── sigmoid.py
  │   │   ├── relu.py
  │   │   └── tanh.py
  │   ├── loss_functions/           # Implementações de funções de perda
  │   │   ├── __init__.py
  │   │   ├── mse.py
  │   │   └── binary_crossentropy.py
  │   └── optimizers/               # Implementações de otimizadores
  │       ├── __init__.py
  │       ├── sgd.py
  │       └── momentum.py
  └── datasets/                    # Módulos para carregamento de datasets
tests/                            # Testes unitários
notebooks/                        # Notebooks Jupyter com exemplos de uso
```

## Requisitos

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (apenas para pré-processamento de dados e avaliação)

## Instalação

```bash
# Criar um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
# venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso Básico

```python
from src.rna import NeuralNetwork

# Criar uma rede neural com 2 entradas, 1 camada oculta com 3 neurônios e 1 saída
nn = NeuralNetwork([2, 3, 1], activation_functions=['sigmoid', 'sigmoid'])

# Treinar a rede neural
nn.fit(X_train, y_train, epochs=1000, learning_rate=0.01, loss_function='mse')

# Fazer predições
predictions = nn.predict(X_test)
```

## Exemplos

Consulte os notebooks na pasta `notebooks/` para exemplos detalhados de:
- Regressão
- Classificação binária
- Classificação multiclasse

## Contribuições

Este projeto foi desenvolvido como parte do trabalho final da disciplina de Machine Learning.

## Autores

- [Nome do Autor 1]
- [Nome do Autor 2]
- [Nome do Autor 3]

## Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo LICENSE para mais detalhes. 