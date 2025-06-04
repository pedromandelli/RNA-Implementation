"""
Implementação do otimizador Gradient Descent com Momentum.
"""

import numpy as np

class Momentum:
    """
    Implementação do otimizador Gradient Descent com Momentum.
    
    O Momentum ajuda a acelerar o SGD nas direções relevantes e
    amortecer oscilações, permitindo convergência mais rápida.
    
    v = γ * v - η * ∇θJ(θ)
    θ = θ + v
    
    onde:
    - θ são os parâmetros (pesos e biases)
    - v é a velocidade (momentum)
    - γ (gamma) é o coeficiente de momentum
    - η (eta) é a taxa de aprendizado
    - ∇θJ(θ) é o gradiente da função de perda com relação aos parâmetros
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Inicializa o otimizador Momentum.
        
        Args:
            learning_rate (float): Taxa de aprendizado
            momentum (float): Coeficiente de momentum (geralmente entre 0.5 e 0.99)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, params, grads):
        """
        Atualiza os parâmetros usando o algoritmo Momentum.
        
        Args:
            params (dict): Dicionário contendo os parâmetros (pesos e biases)
            grads (dict): Dicionário contendo os gradientes dos parâmetros
            
        Returns:
            dict: Parâmetros atualizados
        """
        # Inicializa velocidades se não existirem
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])
        
        # Para cada camada, atualiza as velocidades e os parâmetros
        for key in params:
            # Atualiza a velocidade (v = γ * v - η * ∇θJ(θ))
            self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[key]
            
            # Atualiza os parâmetros (θ = θ + v)
            params[key] += self.velocities[key]
        
        return params 