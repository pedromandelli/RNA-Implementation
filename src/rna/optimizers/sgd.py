"""
Implementação do otimizador Stochastic Gradient Descent (SGD).
"""

import numpy as np

class SGD:
    """
    Implementação do otimizador Stochastic Gradient Descent (SGD).
    
    O SGD atualiza os parâmetros na direção do gradiente negativo da função
    de perda com relação a esses parâmetros, multiplicados pela taxa de aprendizado.
    
    θ = θ - η * ∇θJ(θ)
    
    onde:
    - θ são os parâmetros (pesos e biases)
    - η (eta) é a taxa de aprendizado
    - ∇θJ(θ) é o gradiente da função de perda com relação aos parâmetros
    """
    
    def __init__(self, learning_rate=0.01):
        """
        Inicializa o otimizador SGD.
        
        Args:
            learning_rate (float): Taxa de aprendizado
        """
        self.learning_rate = learning_rate
    
    def update(self, params, grads):
        """
        Atualiza os parâmetros usando o algoritmo SGD.
        
        Args:
            params (dict): Dicionário contendo os parâmetros (pesos e biases)
            grads (dict): Dicionário contendo os gradientes dos parâmetros
            
        Returns:
            dict: Parâmetros atualizados
        """
        # Para cada camada, atualiza os pesos e biases
        for key in params:
            params[key] -= self.learning_rate * grads[key]
        
        return params 