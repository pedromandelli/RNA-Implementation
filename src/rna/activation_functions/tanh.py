"""
Implementação da função de ativação Tangente Hiperbólica (Tanh).
"""

import numpy as np

class Tanh:
    """
    Implementação da função de ativação Tangente Hiperbólica (Tanh).
    
    A função Tanh mapeia qualquer número real para um valor entre -1 e 1,
    sendo útil quando os dados precisam ser centrados em torno de zero.
    
    f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    f'(x) = 1 - tanh^2(x)
    """
    
    @staticmethod
    def forward(Z):
        """
        Aplica a função de ativação Tanh.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Saída da função de ativação
        """
        return np.tanh(Z)
    
    @staticmethod
    def backward(Z):
        """
        Calcula a derivada da função Tanh.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Derivada da função de ativação
        """
        # A derivada da tanh é 1 - tanh^2(x)
        return 1 - np.power(np.tanh(Z), 2) 