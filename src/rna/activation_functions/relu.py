"""
Implementação da função de ativação ReLU (Rectified Linear Unit).
"""

import numpy as np

class ReLU:
    """
    Implementação da função de ativação ReLU (Rectified Linear Unit).
    
    A função ReLU retorna zero para entradas negativas e a própria entrada
    para valores positivos, ajudando a mitigar o problema de desaparecimento
    de gradientes em redes neurais profundas.
    
    f(x) = max(0, x)
    f'(x) = 1 se x > 0, 0 caso contrário
    """
    
    @staticmethod
    def forward(Z):
        """
        Aplica a função de ativação ReLU.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Saída da função de ativação
        """
        return np.maximum(0, Z)
    
    @staticmethod
    def backward(Z):
        """
        Calcula a derivada da função ReLU.
        
        Args:
            Z (numpy.ndarray): Entrada da função de ativação
                
        Returns:
            numpy.ndarray: Derivada da função de ativação
        """
        # A derivada é 1 para entradas positivas e 0 para entradas negativas
        return np.where(Z > 0, 1, 0) 