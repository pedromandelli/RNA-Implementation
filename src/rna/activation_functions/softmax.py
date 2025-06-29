"""
Implementação da função de ativação Softmax.
"""

import numpy as np

class Softmax:
    """
    Implementação da função de ativação Softmax.

    A função Softmax é usada na camada de saída de redes neurais para
    problemas de classificação multiclasse. Ela transforma um vetor de
    valores reais em uma distribuição de probabilidade.

    f(x_i) = e^(x_i) / sum(e^(x_j)) para todo j
    """

    @staticmethod
    def forward(Z):
        """
        Aplica a função de ativação Softmax.

        Args:
            Z (numpy.ndarray): Entrada da função de ativação

        Returns:
            numpy.ndarray: Saída da função de ativação (probabilidades)
        """
        # Subtrair o máximo de Z para estabilidade numérica e evitar overflow
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    @staticmethod
    def backward(Z):
        """
        Calcula a derivada da função Softmax.

        Nota: A derivada da Softmax é uma matriz Jacobiana. No entanto,
        quando usada com a perda de Entropia Cruzada Categórica, a derivada
        combinada (loss + softmax) se simplifica enormemente. Por isso,
        a lógica de backpropagation para essa combinação é tratada de forma
        especial no método `backward` da classe NeuralNetwork, e este
        método de derivada raramente é chamado diretamente.
        """
        # Este método não é tipicamente usado, pois o gradiente
        # dL/dZ é calculado de forma mais simples quando combinado com a
        # entropia cruzada. Retornamos 1 como um placeholder.
        return np.ones_like(Z)