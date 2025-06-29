"""
Funções de perda para a Rede Neural Artificial.
"""

from .mse import MSE
from .binary_crossentropy import BinaryCrossentropy
from .categorical_crossentropy import CategoricalCrossentropy 

__all__ = ['MSE', 'BinaryCrossentropy', 'CategoricalCrossentropy'] 