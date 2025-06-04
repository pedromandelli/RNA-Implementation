"""
Implementação da função de perda Mean Squared Error (MSE).
"""

import numpy as np

class MSE:
    """
    Implementação da função de perda Mean Squared Error (MSE).
    
    O MSE calcula a média dos quadrados das diferenças entre os valores
    previstos e os valores reais. É comumente usado em problemas de regressão.
    
    L(y, y_hat) = (1/n) * sum((y - y_hat)^2)
    dL/dy_hat = (2/n) * (y_hat - y)
    """
    
    @staticmethod
    def forward(y_true, y_pred):
        """
        Calcula a perda Mean Squared Error.
        
        Args:
            y_true (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            y_pred (numpy.ndarray): Valores previstos, shape (n_samples, n_outputs)
                
        Returns:
            float: Valor da perda MSE
        """
        # Garante que os arrays têm a mesma forma
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calcula o MSE
        m = y_true.shape[0]  # número de exemplos
        mse = np.mean(np.square(y_pred - y_true))
        return mse
    
    @staticmethod
    def backward(y_true, y_pred):
        """
        Calcula a derivada da função de perda MSE.
        
        Args:
            y_true (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            y_pred (numpy.ndarray): Valores previstos, shape (n_samples, n_outputs)
                
        Returns:
            numpy.ndarray: Derivada da função de perda MSE
        """
        # Garante que os arrays têm a mesma forma
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calcula a derivada do MSE
        m = y_true.shape[0]  # número de exemplos
        return (2/m) * (y_pred - y_true) 