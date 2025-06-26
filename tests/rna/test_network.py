"""
Testes para a classe NeuralNetwork.
"""

import sys
import os
import numpy as np
import unittest

# Adicionar o diretório src ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.rna import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    """
    Testes para verificar a implementação da classe NeuralNetwork.
    """
    
    def test_initialization(self):
        """Testa a inicialização da rede neural."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Verificar se os pesos e biases foram inicializados com as dimensões corretas
        self.assertEqual(nn.weights[1].shape, (3, 2))
        self.assertEqual(nn.weights[2].shape, (1, 3))
        self.assertEqual(nn.biases[1].shape, (1, 3))
        self.assertEqual(nn.biases[2].shape, (1, 1))
    
    def test_forward(self):
        """Testa a propagação para frente."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Dados de teste
        X = np.array([[0.5, 0.6], [0.1, 0.2]])
        
        # Forward pass
        output = nn.forward(X)
        
        # Verificar dimensões da saída
        self.assertEqual(output.shape, (2, 1))
        
        # Verificar se o cache foi preenchido corretamente
        self.assertTrue('A0' in nn.cache)
        self.assertTrue('Z1' in nn.cache)
        self.assertTrue('A1' in nn.cache)
        self.assertTrue('Z2' in nn.cache)
        self.assertTrue('A2' in nn.cache)
    
    def test_backward(self):
        """Testa a retropropagação."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Dados de teste
        X = np.array([[0.5, 0.6], [0.1, 0.2]])
        y = np.array([[1], [0]])
        
        # Calcular gradientes
        gradients = nn.backward(X, y, loss_function='mse')
        
        # Verificar se os gradientes têm as dimensões corretas
        self.assertEqual(gradients['dW1'].shape, nn.weights[1].shape)
        self.assertEqual(gradients['db1'].shape, nn.biases[1].shape)
        self.assertEqual(gradients['dW2'].shape, nn.weights[2].shape)
        self.assertEqual(gradients['db2'].shape, nn.biases[2].shape)
    
    def test_train_step(self):
        """Testa um passo de treinamento."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Dados de teste
        X = np.array([[0.5, 0.6], [0.1, 0.2]])
        y = np.array([[1], [0]])
        
        # Salvar pesos iniciais
        initial_weights = {}
        for l in range(1, nn.num_layers):
            initial_weights[f'W{l}'] = nn.weights[l].copy()
        
        # Executar um passo de treinamento
        loss = nn.train_step(X, y, learning_rate=0.1, loss_function='mse')
        
        # Verificar se os pesos foram atualizados
        for l in range(1, nn.num_layers):
            self.assertFalse(np.array_equal(nn.weights[l], initial_weights[f'W{l}']))
        
        # Verificar se a perda é um número
        self.assertIsInstance(loss, float)
    
    def test_activation_functions(self):
        """Testa diferentes funções de ativação."""
        # Criar redes com diferentes funções de ativação
        nn_sigmoid = NeuralNetwork([2, 3, 1], activation_functions=['sigmoid', 'sigmoid'])
        nn_relu = NeuralNetwork([2, 3, 1], activation_functions=['relu', 'sigmoid'])
        nn_tanh = NeuralNetwork([2, 3, 1], activation_functions=['tanh', 'sigmoid'])
        
        # Dados de teste
        X = np.array([[0.5, 0.6], [0.1, 0.2]])
        
        # Forward pass com cada rede
        output_sigmoid = nn_sigmoid.forward(X)
        output_relu = nn_relu.forward(X)
        output_tanh = nn_tanh.forward(X)
        
        # Verificar se as saídas têm as dimensões corretas
        self.assertEqual(output_sigmoid.shape, (2, 1))
        self.assertEqual(output_relu.shape, (2, 1))
        self.assertEqual(output_tanh.shape, (2, 1))
        
        # Verificar se as ativações intermediárias são diferentes
        # (indicando que as diferentes funções de ativação foram aplicadas)
        self.assertFalse(np.array_equal(nn_sigmoid.cache['A1'], nn_relu.cache['A1']))
        self.assertFalse(np.array_equal(nn_sigmoid.cache['A1'], nn_tanh.cache['A1']))
        self.assertFalse(np.array_equal(nn_relu.cache['A1'], nn_tanh.cache['A1']))
    
    def test_loss_functions(self):
        """Testa diferentes funções de perda."""
        nn = NeuralNetwork([2, 3, 1])
        
        # Dados de teste
        X = np.array([[0.5, 0.6], [0.1, 0.2]])
        y = np.array([[1], [0]])
        
        # Calcular perda com MSE
        loss_mse = nn.calculate_loss(X, y, loss_function='mse')
        
        # Calcular perda com Binary Cross-Entropy
        loss_bce = nn.calculate_loss(X, y, loss_function='binary_crossentropy')
        
        # Verificar se as perdas são números
        self.assertIsInstance(loss_mse, float)
        self.assertIsInstance(loss_bce, float)
        
        # As perdas devem ser diferentes
        self.assertNotEqual(loss_mse, loss_bce)
    
    def test_fit_basic(self):
        """Testa o método fit em um problema simples."""
        # Criar um problema XOR para testar
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Criar a rede neural
        nn = NeuralNetwork([2, 4, 1], activation_functions=['sigmoid', 'sigmoid'])
        
        # Treinar a rede por algumas épocas
        history = nn.fit(X, y, epochs=1000, learning_rate=0.5, batch_size=4, 
                         loss_function='binary_crossentropy', verbose=False)
        
        # Verificar se o histórico contém as perdas
        self.assertTrue('loss' in history)
        self.assertEqual(len(history['loss']), 1000)
        
        # Verificar se a perda diminuiu durante o treinamento
        self.assertLess(history['loss'][-1], history['loss'][0])
        
        # Fazer predições
        predictions = nn.predict(X, output_activation='sigmoid')
        
        # Verificar se as predições estão no formato correto
        self.assertEqual(predictions.shape, (4, 1))
        
        # Converter predições para binário
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Verificar a acurácia (deve ser razoavelmente boa após o treinamento)
        accuracy = np.mean(binary_predictions == y)
        print(f"XOR accuracy: {accuracy}")
        
        # A acurácia deve ser pelo menos 75% (3/4 corretos)
        self.assertGreaterEqual(accuracy, 0.75)

if __name__ == '__main__':
    unittest.main() 