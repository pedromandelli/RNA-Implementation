"""
Módulo principal para a implementação da Rede Neural Artificial.
"""
import numpy as np
from .activation import ACTIVATION_FUNCTIONS


class NeuralNetwork:
    """
    Implementação de uma Rede Neural Artificial.
    
    Esta classe implementa uma rede neural artificial multicamada com
    funcionalidades para configuração da arquitetura, treinamento e
    predição.
    """
    
    def __init__(self, layer_sizes, init_method='xavier', activation='relu'):
        """
        Inicializa uma rede neural com a arquitetura especificada.
        
        Args:
            layer_sizes (list): Lista contendo o número de neurônios em cada camada,
                              incluindo camada de entrada e saída.
                              Ex: [3, 4, 2, 1] representa uma rede com 3 entradas,
                              2 camadas ocultas (4 e 2 neurônios) e 1 saída.
            init_method (str): Método de inicialização dos pesos. Opções: 'xavier', 'he', 'random'.
            activation (str): Função de ativação para as camadas ocultas. Opções: 'relu', 'sigmoid', 'linear'.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activation = activation
        
        # Verifica se a função de ativação está disponível
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Função de ativação '{activation}' não implementada")
        
        # Obtém as funções de ativação e suas derivadas
        self.activation_func, self.activation_derivative = ACTIVATION_FUNCTIONS[activation]
        
        # Dicionários para armazenar pesos e biases
        self.weights = {}
        self.biases = {}
        
        # Dicionário para armazenar valores intermediários durante a propagação
        # (necessário para o backpropagation)
        self.cache = {}
        
        # Inicializa os pesos e biases
        self.initialize_weights(method=init_method)
    
    def initialize_weights(self, method='xavier'):
        """
        Inicializa os pesos da rede neural.
        
        Args:
            method (str): Método de inicialização. Opções: 'xavier', 'he', 'random'
            
        Returns:
            tuple: (weights, biases) - Dicionários com os pesos e biases inicializados
        """
        for l in range(1, self.num_layers):
            if method == 'xavier':
                # Inicialização Xavier/Glorot
                # Boa para funções de ativação como tanh e sigmoid
                bound = np.sqrt(6) / np.sqrt(self.layer_sizes[l-1] + self.layer_sizes[l])
                self.weights[l] = np.random.uniform(-bound, bound, (self.layer_sizes[l], self.layer_sizes[l-1]))
            elif method == 'he':
                # Inicialização He
                # Boa para funções de ativação ReLU
                self.weights[l] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
            else:
                # Inicialização aleatória simples
                self.weights[l] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            
            # Inicializar biases com zeros
            self.biases[l] = np.zeros((self.layer_sizes[l], 1))
        
        return self.weights, self.biases
    
    def forward(self, X):
        """
        Realiza a propagação para frente.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
                
        Returns:
            numpy.ndarray: Saída da rede neural
        """
        # Garantir que X tenha o formato correto (n_samples, n_features)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Verificar se o número de features corresponde ao esperado
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Número de features ({X.shape[1]}) não corresponde ao esperado ({self.layer_sizes[0]})")
        
        # Armazenar a entrada no cache para backpropagation
        self.cache = {}
        self.cache['A0'] = X
        
        # Propagar através de cada camada
        A = X  # Ativação da camada de entrada
        
        for l in range(1, self.num_layers):
            # Calcular entrada ponderada (Z) para a camada l
            Z = np.dot(A, self.weights[l].T) + self.biases[l].T
            
            # Armazenar Z no cache para backpropagation
            self.cache[f'Z{l}'] = Z
            
            # Aplicar função de ativação
            if l < self.num_layers - 1:  # Camadas ocultas
                A = self.activation_func(Z)
            else:  # Camada de saída (linear por enquanto)
                A = Z
            
            # Armazenar ativação no cache
            self.cache[f'A{l}'] = A
        
        # Retornar a ativação da última camada
        return A
    
    def predict(self, X):
        """
        Realiza predições usando a rede neural treinada.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predições da rede neural
        """
        # Utiliza o método forward
        return self.forward(X)
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, batch_size=None, verbose=True):
        """
        Treina a rede neural usando o algoritmo de gradiente descendente.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            y (numpy.ndarray): Valores alvo, shape (n_samples, n_outputs)
            epochs (int): Número de épocas de treinamento
            learning_rate (float): Taxa de aprendizado
            batch_size (int, optional): Tamanho do lote para mini-batch GD
            verbose (bool): Se True, exibe progresso do treinamento
            
        Returns:
            list: Histórico de valores da função de perda
        """
        # A implementação será feita em TASK011
        pass
    
    def __str__(self):
        """Retorna uma representação em string da arquitetura da rede."""
        return f"NeuralNetwork(layers={self.layer_sizes}, activation={self.activation})" 