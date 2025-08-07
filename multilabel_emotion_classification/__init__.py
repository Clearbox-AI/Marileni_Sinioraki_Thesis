"""
Multilabel Emotion Classification Package

A modular implementation for training and evaluating multilabel emotion classification models
on the GoEmotions dataset using BERT-based transformers.
"""

__version__ = "1.0.0"
__author__ = "Marileni Sinioraki"

from . import data
from . import preprocessing  
from . import models
from . import training
from . import evaluation
from . import config
from . import utils

__all__ = [
    "data",
    "preprocessing", 
    "models",
    "training",
    "evaluation",
    "config",
    "utils"
]
