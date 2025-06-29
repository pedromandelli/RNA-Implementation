"""
Módulo principal para a implementação da Rede Neural Artificial.
"""
import numpy as np
from .activation_functions import Sigmoid, ReLU, Tanh, Softmax
from .loss_functions import MSE, BinaryCrossentropy, CategoricalCrossentropy
from .optimizers import SGD, Momentum


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
        
        # Mapear as strings de ativação para as classes correspondentes
        self.activation_map = {
            'sigmoid': Sigmoid,
            'relu': ReLU,
            'tanh': Tanh,
            'softmax': Softmax,
            'linear': None  # Linear não precisa de classe específica
        }
        
        # Mapear as strings de função de perda para as classes correspondentes
        self.loss_map = {
            'mse': MSE,
            'binary_crossentropy': BinaryCrossentropy,
            'categorical_crossentropy': CategoricalCrossentropy
        }
        
        # Mapear as strings de otimizador para as classes correspondentes
        self.optimizer_map = {
            'sgd': SGD,
            'momentum': Momentum
        }
    
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
                A = self.activation_map['sigmoid'].forward(Z)
            elif activation == 'relu':
                A = self.activation_map['relu'].forward(Z)
            elif activation == 'tanh':
                A = self.activation_map['tanh'].forward(Z)
            elif activation == 'softmax':
                A = self.activation_map['softmax'].forward(Z)
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
    
    def backward(self, X, y, loss_function='mse', output_activation=None):
        """
        Realiza a retropropagação (backpropagation) para calcular gradientes.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            y (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            loss_function (str): Função de perda a ser usada ('mse' ou 'binary_crossentropy')
            output_activation (str, optional): Ativação da camada de saída.
                
        Returns:
            dict: Gradientes dos pesos e biases
        """
        # Define a ativação da saída com base na função de perda, se não especificada
        if output_activation is None:
            if loss_function == 'binary_crossentropy':
                output_activation = 'sigmoid'
            else:
                output_activation = 'linear'
        
        # Forward pass para preencher o cache
        y_pred = self.forward(X, output_activation=output_activation)
        
        # Inicializar gradientes
        m = X.shape[0]  # número de exemplos
        gradients = {}
        
        # Calcular derivada da função de perda
        if loss_function == 'mse':
            if output_activation == 'sigmoid':
                # Se usamos sigmoid na saída com MSE, precisamos considerar a derivada da sigmoid
                dZ = self.loss_map['mse'].backward(y, y_pred) * self.activation_map['sigmoid'].backward(self.cache[f'Z{self.num_layers - 1}'])
            else:
                dZ = self.loss_map['mse'].backward(y, y_pred)
        elif loss_function == 'binary_crossentropy':
            if output_activation == 'sigmoid':
                # Para BCE com sigmoid, a derivada se simplifica para y_pred - y
                dZ = self.loss_map['binary_crossentropy'].backward_with_sigmoid(y, y_pred)
            else:
                dZ = self.loss_map['binary_crossentropy'].backward(y, y_pred)
        elif loss_function == 'categorical_crossentropy': 
                if output_activation == 'softmax':
                    # Para CCE com Softmax, a derivada se simplifica para y_pred - y
                    dZ = self.loss_map['categorical_crossentropy'].backward_with_softmax(y, y_pred)
                else:
                    raise NotImplementedError("Categorical Crossentropy sem Softmax na saída não é diretamente suportado pela forma simplificada.")            
        else:
            raise ValueError(f"Função de perda '{loss_function}' não implementada")
        
        # Backpropagation para a última camada
        A_prev = self.cache[f'A{self.num_layers - 2}']
        gradients[f'dW{self.num_layers - 1}'] = np.dot(dZ.T, A_prev) / m
        gradients[f'db{self.num_layers - 1}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Backpropagation para as camadas ocultas
        for l in reversed(range(1, self.num_layers - 1)):
            # Calcular dZ para a camada l
            dA = np.dot(dZ, self.weights[l+1])
            Z = self.cache[f'Z{l}']
            
            # Aplicar derivada da função de ativação
            activation = self.activation_functions[l-1]
            if activation == 'sigmoid':
                dZ = dA * self.activation_map['sigmoid'].backward(Z)
            elif activation == 'relu':
                dZ = dA * self.activation_map['relu'].backward(Z)
            elif activation == 'tanh':
                dZ = dA * self.activation_map['tanh'].backward(Z)
            else:  # linear
                dZ = dA
            
            # Calcular gradientes para a camada l
            if l > 1:
                A_prev = self.cache[f'A{l-1}']
            else:
                A_prev = X
            
            gradients[f'dW{l}'] = np.dot(dZ.T, A_prev) / m
            gradients[f'db{l}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        return gradients
    
    def update_params(self, gradients, optimizer, learning_rate=0.01, **kwargs):
        """
        Atualiza os parâmetros da rede usando o otimizador especificado.
        
        Args:
            gradients (dict): Gradientes calculados pelo método backward
            optimizer (str ou objeto): Otimizador a ser usado ('sgd' ou 'momentum') ou instância de otimizador
            learning_rate (float): Taxa de aprendizado
            **kwargs: Argumentos adicionais para o otimizador
            
        Returns:
            None
        """
        # Se optimizer for uma string, criar uma instância do otimizador correspondente
        if isinstance(optimizer, str):
            if optimizer.lower() == 'sgd':
                optimizer_instance = self.optimizer_map['sgd'](learning_rate=learning_rate)
            elif optimizer.lower() == 'momentum':
                momentum = kwargs.get('momentum', 0.9)
                optimizer_instance = self.optimizer_map['momentum'](learning_rate=learning_rate, momentum=momentum)
            else:
                raise ValueError(f"Otimizador '{optimizer}' não implementado")
        else:
            # Se já for uma instância de otimizador, usar diretamente
            optimizer_instance = optimizer
        
        # Preparar dicionários de parâmetros para o otimizador
        params = {}
        grads = {}
        
        # Organizar pesos e biases para o otimizador
        for l in range(1, self.num_layers):
            params[f'W{l}'] = self.weights[l]
            params[f'b{l}'] = self.biases[l]
            grads[f'W{l}'] = gradients[f'dW{l}']
            grads[f'b{l}'] = gradients[f'db{l}']
        
        # Atualizar parâmetros usando o otimizador
        updated_params = optimizer_instance.update(params, grads)
        
        # Atualizar pesos e biases da rede
        for l in range(1, self.num_layers):
            self.weights[l] = updated_params[f'W{l}']
            self.biases[l] = updated_params[f'b{l}']
    
    def train_step(self, X, y, learning_rate=0.01, loss_function='mse', 
                   output_activation=None, optimizer='sgd', **kwargs):
        """
        Realiza um passo de treinamento (forward + backward + atualização de pesos).
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            y (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            learning_rate (float): Taxa de aprendizado
            loss_function (str): Função de perda a ser usada ('mse' ou 'binary_crossentropy')
            output_activation (str, optional): Ativação da camada de saída.
            optimizer (str): Tipo de otimizador ('sgd' ou 'momentum')
            **kwargs: Argumentos adicionais para o otimizador
                
        Returns:
            float: Valor da perda após o passo de treinamento
        """
        # Calcular gradientes com backpropagation
        gradients = self.backward(X, y, loss_function, output_activation)
        
        # Atualizar pesos e biases
        self.update_params(gradients, optimizer, learning_rate, **kwargs)
        
        # Calcular e retornar a perda após a atualização
        return self.calculate_loss(X, y, loss_function, output_activation)
    
    def fit(self, X, y, epochs=1000, batch_size=32, learning_rate=0.01, loss_function='mse', 
            output_activation=None, optimizer='sgd', verbose=True, validation_data=None, **kwargs):
        """
        Treina a rede neural.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            y (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            epochs (int): Número de épocas
            batch_size (int): Tamanho do mini-batch
            learning_rate (float): Taxa de aprendizado
            loss_function (str): Função de perda a ser usada ('mse' ou 'binary_crossentropy')
            output_activation (str, optional): Ativação da camada de saída.
            optimizer (str): Tipo de otimizador ('sgd' ou 'momentum')
            verbose (bool): Se True, imprime progresso do treinamento
            validation_data (tuple): (X_val, y_val) para validação durante o treinamento
            **kwargs: Argumentos adicionais para o otimizador
                
        Returns:
            dict: Histórico de treinamento
        """
        if output_activation is None:
            output_activation = self.activation_functions[-1]
            
        history = {'loss': [], 'val_loss': []}
        m = X.shape[0]  # número de exemplos
        
        for epoch in range(epochs):
            # Embaralhar os dados
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Realizar um passo de treinamento
                loss = self.train_step(
                    X_batch, y_batch, 
                    learning_rate, 
                    loss_function, 
                    output_activation, 
                    optimizer,
                    **kwargs
                )
            
            # Calcular perda para a época completa
            epoch_loss = self.calculate_loss(X, y, loss_function, output_activation)
            history['loss'].append(epoch_loss)
            
            # Calcular perda de validação se fornecida
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self.calculate_loss(X_val, y_val, loss_function, output_activation)
                history['val_loss'].append(val_loss)
            
            # Imprimir progresso
            if verbose and (epoch + 1) % 100 == 0:
                val_str = ""
                if validation_data is not None:
                    val_str = f" - val_loss: {history['val_loss'][-1]:.4f}"
                    
                print(f"Época {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}{val_str}")
        
        return history
    
    def calculate_loss(self, X, y, loss_function='mse', output_activation=None):
        """
        Calcula a perda para um conjunto de dados.
        
        Args:
            X (numpy.ndarray): Dados de entrada, shape (n_samples, n_features)
            y (numpy.ndarray): Valores reais, shape (n_samples, n_outputs)
            loss_function (str): Função de perda a ser usada ('mse' ou 'binary_crossentropy')
            output_activation (str, optional): Ativação da camada de saída.
                
        Returns:
            float: Valor da perda
        """
        # Define a ativação da saída com base na função de perda, se não especificada
        if output_activation is None:
            if loss_function == 'binary_crossentropy':
                output_activation = 'sigmoid'
            else:
                output_activation = 'linear'
        
        # Forward pass para obter previsões
        y_pred = self.forward(X, output_activation=output_activation)
        
        # Calcular perda usando a função de perda especificada
        if loss_function == 'mse':
            return self.loss_map['mse'].forward(y, y_pred)
        elif loss_function == 'binary_crossentropy':
            return self.loss_map['binary_crossentropy'].forward(y, y_pred)
        elif loss_function == 'categorical_crossentropy':
            return self.loss_map['categorical_crossentropy'].forward(y, y_pred)
        else:
            raise ValueError(f"Função de perda '{loss_function}' não implementada")
    
    def __str__(self):
        """Retorna uma representação em string da arquitetura da rede."""
        return f"NeuralNetwork(layers={self.layer_sizes}, activation={self.activation_functions})" 