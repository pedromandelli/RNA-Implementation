"""
Implementação da função de perda Categorical Cross-Entropy.
"""

import numpy as np

class CategoricalCrossentropy:
    """
    Implementação da função de perda Categorical Cross-Entropy.

    Esta função de perda é usada para problemas de classificação multiclasse.
    Requer que os rótulos verdadeiros (y_true) estejam no formato one-hot encoded.

    L(y, y_hat) = -(1/n) * sum(y * log(y_hat))
    """

    @staticmethod
    def forward(y_true, y_pred):
        """
        Calcula a perda Categorical Cross-Entropy.

        Args:
            y_true (numpy.ndarray): Valores reais (one-hot encoded), shape (n_samples, n_classes)
            y_pred (numpy.ndarray): Valores previstos (probabilidades), shape (n_samples, n_classes)

        Returns:
            float: Valor da perda
        """
        # Adiciona um epsilon pequeno para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        m = y_true.shape[0]  # número de exemplos
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    @staticmethod
    def backward_with_softmax(y_true, y_pred):
        """
        Calcula a derivada da CCE quando usada com Softmax.

        A derivada combinada se simplifica para y_pred - y_true.

        Args:
            y_true (numpy.ndarray): Valores reais (one-hot encoded)
            y_pred (numpy.ndarray): Valores previstos (probabilidades da Softmax)

        Returns:
            numpy.ndarray: Gradiente dL/dZ
        """
        return y_pred - y_true