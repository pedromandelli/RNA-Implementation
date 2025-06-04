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
    
    def __init__(self, layer_sizes, activation_functions=None):
        """
        Inicializa uma rede neural com a arquitetura especificada.
        
        Args:
            layer_sizes (list): Lista contendo o número de neurônios em cada camada,
                               incluindo camada de entrada e saída.
            activation_functions (list, optional): Lista de funções de ativação para cada camada.
                                                  Valores possíveis: 'sigmoid', 'relu', 'tanh'.
                                                  Default é 'sigmoid' para todas as camadas ocultas.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Inicializar pesos e biases
        self.weights = {}
        self.biases = {}
        self.cache = {}
        
        # Inicialização de Xavier para os pesos
        for l in range(1, self.num_layers):
            # Inicialização de Xavier: adequada para funções de ativação sigmoidais
            self.weights[l] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(1 / layer_sizes[l-1])
            self.biases[l] = np.zeros((1, layer_sizes[l]))
        
        # Configurar funções de ativação
        if activation_functions is None:
            self.activation_functions = ['sigmoid'] * (self.num_layers - 2) + ['linear']
        else:
            self.activation_functions = activation_functions
    
    def forward(self, X, output_activation=None):
        """
        Realiza a propagação para frente.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            output_activation (str, optional): Ativação da camada de saída.
                Se None, usa o valor definido em self.activation_functions.
                
        Returns:
            numpy.ndarray: Saída da rede neural
        """
        self.cache = {}
        A = X
        self.cache['A0'] = A
        
        for l in range(1, self.num_layers):
            # Calcular entrada ponderada (Z) para a camada l
            Z = np.dot(A, self.weights[l].T) + self.biases[l]
            
            # Armazenar Z no cache para backpropagation
            self.cache[f'Z{l}'] = Z
            
            # Determinar a função de ativação para esta camada
            activation = self.activation_functions[l-1]
            if l == self.num_layers - 1 and output_activation is not None:
                activation = output_activation
            
            # Aplicar função de ativação
            if activation == 'sigmoid':
                A = self._sigmoid(Z)
            elif activation == 'relu':
                A = self._relu(Z)
            elif activation == 'tanh':
                A = self._tanh(Z)
            else:  # linear
                A = Z
            
            # Armazenar ativação no cache
            self.cache[f'A{l}'] = A
        
        return A
    
    def predict(self, X, output_activation=None):
        """
        Realiza predições usando a rede neural treinada.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            output_activation (str, optional): Ativação da camada de saída.
                
        Returns:
            numpy.ndarray: Predições da rede neural
        """
        return self.forward(X, output_activation)
    
    # Implementações básicas das funções de ativação - serão substituídas por classes dedicadas
    def _sigmoid(self, Z):
        """Implementação temporária da função sigmoid"""
        return 1 / (1 + np.exp(-Z))
    
    def _relu(self, Z):
        """Implementação temporária da função ReLU"""
        return np.maximum(0, Z)
    
    def _tanh(self, Z):
        """Implementação temporária da função tanh"""
        return np.tanh(Z)
    
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
        return f"NeuralNetwork(layers={self.layer_sizes}, activation={self.activation_functions})" 