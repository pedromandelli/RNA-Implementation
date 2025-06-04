"""
Módulo para funções de ativação da rede neural.

Este módulo contém implementações de várias funções de ativação e suas derivadas.
"""
import numpy as np


def relu(Z):
    """
    Aplica a função de ativação ReLU (Rectified Linear Unit).
    
    Args:
        Z (numpy.ndarray): Entrada da função de ativação
            
    Returns:
        numpy.ndarray: Saída da função de ativação
    """
    return np.maximum(0, Z)


def relu_derivative(Z):
    """
    Calcula a derivada da função ReLU.
    
    Args:
        Z (numpy.ndarray): Entrada da função de ativação
            
    Returns:
        numpy.ndarray: Derivada da função de ativação
    """
    return np.where(Z > 0, 1, 0)


# Dicionário de funções de ativação e suas derivadas
ACTIVATION_FUNCTIONS = {
    'relu': (relu, relu_derivative),
    'linear': (lambda x: x, lambda x: np.ones_like(x))
} 