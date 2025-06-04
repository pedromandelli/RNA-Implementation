"""
Implementação da função de perda Binary Cross-Entropy.
"""

import numpy as np

class BinaryCrossentropy:
    """
    Implementação da função de perda Binary Cross-Entropy.
    
    A Binary Cross-Entropy é usada em problemas de classificação binária
    onde a saída é uma probabilidade entre 0 e 1.
    
    L(y, y_hat) = -(1/n) * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
    dL/dy_hat = -(1/n) * (y/y_hat - (1-y)/(1-y_hat))
    
    Quando usado com sigmoid na camada de saída, a derivada se simplifica para:
    dL/dz = y_hat - y
    """
    
    @staticmethod
    def forward(y_true, y_pred):
        """
        Calcula a perda Binary Cross-Entropy.
        
        Args:
            y_true (numpy.ndarray): Valores reais (0 ou 1), shape (n_samples, n_outputs)
            y_pred (numpy.ndarray): Valores previstos (probabilidades), shape (n_samples, n_outputs)
                
        Returns:
            float: Valor da perda Binary Cross-Entropy
        """
        # Garante que os arrays têm a mesma forma
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Adiciona um epsilon pequeno para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calcula a BCE
        m = y_true.shape[0]  # número de exemplos
        bce = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return bce
    
    @staticmethod
    def backward(y_true, y_pred):
        """
        Calcula a derivada da função de perda Binary Cross-Entropy.
        
        Args:
            y_true (numpy.ndarray): Valores reais (0 ou 1), shape (n_samples, n_outputs)
            y_pred (numpy.ndarray): Valores previstos (probabilidades), shape (n_samples, n_outputs)
                
        Returns:
            numpy.ndarray: Derivada da função de perda BCE
        """
        # Garante que os arrays têm a mesma forma
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Adiciona um epsilon pequeno para evitar divisão por zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calcula a derivada da BCE
        m = y_true.shape[0]  # número de exemplos
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / m
    
    @staticmethod
    def backward_with_sigmoid(y_true, y_pred):
        """
        Calcula a derivada da função de perda Binary Cross-Entropy quando usada com Sigmoid.
        
        Esta é uma forma simplificada que é numericamente mais estável.
        
        Args:
            y_true (numpy.ndarray): Valores reais (0 ou 1), shape (n_samples, n_outputs)
            y_pred (numpy.ndarray): Valores previstos (probabilidades), shape (n_samples, n_outputs)
                
        Returns:
            numpy.ndarray: Derivada da função de perda BCE com Sigmoid
        """
        # Quando a BCE é usada com sigmoid, a derivada se simplifica para y_pred - y_true
        return y_pred - y_true 