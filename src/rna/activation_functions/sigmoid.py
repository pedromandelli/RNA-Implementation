"""
Implementação da função de ativação Sigmoid.
"""

import numpy as np

class Sigmoid:
    """
    Implementação da função de ativação Sigmoid.
    
    A função Sigmoid mapeia qualquer número real para um valor entre 0 e 1,
    sendo útil para outputs que representam probabilidades.
    
    f(x) = 1 / (1 + e^(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    @staticmethod
    def forward(Z):
        """
        Aplica a função de ativação Sigmoid.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Saída da função de ativação
        """
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def backward(Z):
        """
        Calcula a derivada da função Sigmoid.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Derivada da função de ativação
        """
        # Calcula o valor da função sigmoid para Z
        sigmoid_Z = 1 / (1 + np.exp(-Z))
        # A derivada da sigmoid é sigmoid(Z) * (1 - sigmoid(Z))
        return sigmoid_Z * (1 - sigmoid_Z) 