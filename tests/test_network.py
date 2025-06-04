"""
Testes para a classe NeuralNetwork.
"""
import unittest
import numpy as np
import sys
import os

# Adiciona o diretório src ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rna.network import NeuralNetwork
from src.rna.activation import relu, relu_derivative


class TestNeuralNetwork(unittest.TestCase):
    """Testes para a classe NeuralNetwork."""
    
    def test_initialization(self):
        """Testa a inicialização da rede neural."""
        # Cria uma rede com 3 entradas, 2 camadas ocultas (4 e 2 neurônios) e 1 saída
        nn = NeuralNetwork([3, 4, 2, 1])
        
        # Verifica se os atributos foram corretamente inicializados
        self.assertEqual(nn.num_layers, 4)
        self.assertEqual(nn.layer_sizes, [3, 4, 2, 1])
        self.assertIsInstance(nn.weights, dict)
        self.assertIsInstance(nn.biases, dict)
        self.assertIsInstance(nn.cache, dict)
        
        # Verifica se os pesos e biases foram inicializados
        self.assertIn(1, nn.weights)
        self.assertIn(2, nn.weights)
        self.assertIn(3, nn.weights)
        self.assertIn(1, nn.biases)
        self.assertIn(2, nn.biases)
        self.assertIn(3, nn.biases)
        
    def test_str_representation(self):
        """Testa a representação em string da rede neural."""
        nn = NeuralNetwork([3, 4, 2, 1])
        self.assertEqual(str(nn), "NeuralNetwork(layers=[3, 4, 2, 1], activation=relu)")
        
    def test_weight_initialization_xavier(self):
        """Testa a inicialização de pesos usando o método Xavier/Glorot."""
        nn = NeuralNetwork([3, 4, 2, 1], init_method='xavier')
        
        # Verifica as dimensões dos pesos e biases
        self.assertEqual(nn.weights[1].shape, (4, 3))  # Primeira camada oculta
        self.assertEqual(nn.weights[2].shape, (2, 4))  # Segunda camada oculta
        self.assertEqual(nn.weights[3].shape, (1, 2))  # Camada de saída
        
        self.assertEqual(nn.biases[1].shape, (4, 1))
        self.assertEqual(nn.biases[2].shape, (2, 1))
        self.assertEqual(nn.biases[3].shape, (1, 1))
        
        # Verifica se os biases são zeros
        self.assertTrue(np.all(nn.biases[1] == 0))
        self.assertTrue(np.all(nn.biases[2] == 0))
        self.assertTrue(np.all(nn.biases[3] == 0))
        
        # Verifica se os pesos estão dentro dos limites esperados para Xavier
        for layer in range(1, nn.num_layers):
            bound = np.sqrt(6) / np.sqrt(nn.layer_sizes[layer-1] + nn.layer_sizes[layer])
            self.assertTrue(np.all(nn.weights[layer] >= -bound))
            self.assertTrue(np.all(nn.weights[layer] <= bound))
    
    def test_weight_initialization_he(self):
        """Testa a inicialização de pesos usando o método He."""
        nn = NeuralNetwork([3, 4, 2, 1], init_method='he')
        
        # Verifica as dimensões dos pesos e biases
        self.assertEqual(nn.weights[1].shape, (4, 3))
        self.assertEqual(nn.weights[2].shape, (2, 4))
        self.assertEqual(nn.weights[3].shape, (1, 2))
        
        # Verifica se os biases são zeros
        self.assertTrue(np.all(nn.biases[1] == 0))
        self.assertTrue(np.all(nn.biases[2] == 0))
        self.assertTrue(np.all(nn.biases[3] == 0))
        
        # Verifica se os pesos foram inicializados (não são todos zeros)
        for layer in range(1, nn.num_layers):
            self.assertFalse(np.all(nn.weights[layer] == 0))
    
    def test_different_initialization_methods(self):
        """Testa se diferentes métodos de inicialização produzem pesos diferentes."""
        nn1 = NeuralNetwork([3, 4, 2, 1], init_method='xavier')
        nn2 = NeuralNetwork([3, 4, 2, 1], init_method='he')
        nn3 = NeuralNetwork([3, 4, 2, 1], init_method='random')
        
        # Verifica se os pesos são diferentes entre os métodos
        for layer in range(1, nn1.num_layers):
            self.assertFalse(np.array_equal(nn1.weights[layer], nn2.weights[layer]))
            self.assertFalse(np.array_equal(nn1.weights[layer], nn3.weights[layer]))
            self.assertFalse(np.array_equal(nn2.weights[layer], nn3.weights[layer]))
    
    def test_reinitialize_weights(self):
        """Testa a reinicialização de pesos."""
        nn = NeuralNetwork([3, 4, 2, 1])
        
        # Salva os pesos originais
        original_weights = {}
        for layer in range(1, nn.num_layers):
            original_weights[layer] = nn.weights[layer].copy()
        
        # Reinicializa os pesos
        nn.initialize_weights(method='xavier')
        
        # Verifica se os pesos mudaram
        for layer in range(1, nn.num_layers):
            self.assertFalse(np.array_equal(original_weights[layer], nn.weights[layer]))
    
    def test_forward_single_sample(self):
        """Testa a propagação para frente com uma única amostra."""
        # Cria uma rede simples
        nn = NeuralNetwork([2, 3, 1])
        
        # Define pesos e biases manualmente para testes determinísticos
        nn.weights[1] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        nn.weights[2] = np.array([[0.7, 0.8, 0.9]])
        nn.biases[1] = np.array([[0.01], [0.02], [0.03]])
        nn.biases[2] = np.array([[0.04]])
        
        # Cria dados de teste
        X = np.array([0.5, 0.6])
        
        # Executa forward pass
        output = nn.forward(X)
        
        # Verifica se a saída tem a forma correta
        self.assertEqual(output.shape, (1, 1))
        
        # Calcula manualmente o resultado esperado
        # Z1 = X @ W1.T + b1.T
        Z1 = np.dot(X, nn.weights[1].T) + nn.biases[1].T
        A1 = Z1  # Sem função de ativação no momento
        # Z2 = A1 @ W2.T + b2.T
        Z2 = np.dot(A1, nn.weights[2].T) + nn.biases[2].T
        expected_output = Z2
        
        # Verifica se o resultado está correto
        np.testing.assert_allclose(output, expected_output)
        
        # Verifica se o cache foi preenchido corretamente
        self.assertIn('A0', nn.cache)
        self.assertIn('Z1', nn.cache)
        self.assertIn('A1', nn.cache)
        self.assertIn('Z2', nn.cache)
        self.assertIn('A2', nn.cache)
        
        # Verifica dimensões dos valores no cache
        self.assertEqual(nn.cache['A0'].shape, (1, 2))
        self.assertEqual(nn.cache['Z1'].shape, (1, 3))
        self.assertEqual(nn.cache['A1'].shape, (1, 3))
        self.assertEqual(nn.cache['Z2'].shape, (1, 1))
        self.assertEqual(nn.cache['A2'].shape, (1, 1))
    
    def test_forward_batch(self):
        """Testa a propagação para frente com um lote de amostras."""
        # Cria uma rede simples
        nn = NeuralNetwork([2, 3, 1])
        
        # Define pesos e biases manualmente para testes determinísticos
        nn.weights[1] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        nn.weights[2] = np.array([[0.7, 0.8, 0.9]])
        nn.biases[1] = np.array([[0.01], [0.02], [0.03]])
        nn.biases[2] = np.array([[0.04]])
        
        # Cria dados de teste (2 amostras)
        X = np.array([[0.5, 0.6], [0.7, 0.8]])
        
        # Executa forward pass
        output = nn.forward(X)
        
        # Verifica se a saída tem a forma correta
        self.assertEqual(output.shape, (2, 1))
        
        # Verifica dimensões dos valores no cache
        self.assertEqual(nn.cache['A0'].shape, (2, 2))
        self.assertEqual(nn.cache['Z1'].shape, (2, 3))
        self.assertEqual(nn.cache['A1'].shape, (2, 3))
        self.assertEqual(nn.cache['Z2'].shape, (2, 1))
        self.assertEqual(nn.cache['A2'].shape, (2, 1))
    
    def test_forward_invalid_input(self):
        """Testa a propagação para frente com entrada inválida."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Testa com número incorreto de features
        with self.assertRaises(ValueError):
            nn.forward(np.array([0.5, 0.6, 0.7]))
    
    def test_activation_functions(self):
        """Testa as funções de ativação."""
        # Testa a função ReLU
        Z = np.array([[-1.0, 0.0, 2.0], [3.0, -4.0, 0.5]])
        A = relu(Z)
        
        # Valores esperados
        expected = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 0.5]])
        
        # Verifica se o resultado é correto
        np.testing.assert_allclose(A, expected)
        
        # Testa a derivada da ReLU
        dA = relu_derivative(Z)
        expected_derivative = np.array([[0, 0, 1], [1, 0, 1]])
        np.testing.assert_allclose(dA, expected_derivative)
    
    def test_forward_with_relu(self):
        """Testa a propagação para frente com função de ativação ReLU."""
        # Cria uma rede com função de ativação ReLU
        nn = NeuralNetwork([2, 3, 1], activation='relu')
        
        # Define pesos e biases manualmente para testes determinísticos
        nn.weights[1] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        nn.weights[2] = np.array([[0.7, 0.8, 0.9]])
        nn.biases[1] = np.array([[0.01], [0.02], [0.03]])
        nn.biases[2] = np.array([[0.04]])
        
        # Cria dados de teste que produzirão valores negativos na camada oculta
        X = np.array([[-0.5, -0.6], [-0.7, -0.8]])
        
        # Executa forward pass
        output = nn.forward(X)
        
        # Verifica se a ativação ReLU foi aplicada na camada oculta
        Z1 = nn.cache['Z1']
        A1 = nn.cache['A1']
        expected_A1 = np.maximum(0, Z1)  # ReLU
        np.testing.assert_allclose(A1, expected_A1)
        
        # Verifica se a camada de saída usa ativação linear
        Z2 = nn.cache['Z2']
        A2 = nn.cache['A2']
        np.testing.assert_allclose(A2, Z2)  # Linear (sem ativação)
    
    def test_activation_validation(self):
        """Testa a validação da função de ativação."""
        # Testa função de ativação válida
        nn = NeuralNetwork([2, 3, 1], activation='relu')
        self.assertEqual(nn.activation, 'relu')
        
        # Testa função de ativação inválida
        with self.assertRaises(ValueError):
            nn = NeuralNetwork([2, 3, 1], activation='invalid_activation')


if __name__ == '__main__':
    unittest.main() 