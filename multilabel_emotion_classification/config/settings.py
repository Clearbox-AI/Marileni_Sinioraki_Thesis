"""Configuration settings and constants."""

import os
from typing import List, Dict, Any

# GoEmotions emotion labels
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Mapping of emotion labels to indices
EMOTION_LABEL_TO_ID = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID_TO_EMOTION_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

# Default paths
DEFAULT_DATA_PATH = "./data"
DEFAULT_OUTPUT_PATH = "./output"
DEFAULT_MODEL_PATH = "./models"
DEFAULT_CONFIG_PATH = "./config.json"

# Data file names
GOEMOTIONS_FILES = [
    "goemotions_1.csv",
    "goemotions_2.csv", 
    "goemotions_3.csv"
]

# Model configurations
SUPPORTED_MODELS = [
    "bert-base-uncased",
    "bert-base-cased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2"
]

# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "max_length": 512,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "dropout_rate": 0.1
}

# Augmentation settings
DEFAULT_AUGMENTATION = {
    "eda": {
        "alpha_sr": 0.1,  # synonym replacement
        "alpha_ri": 0.1,  # random insertion
        "alpha_rs": 0.1,  # random swap
        "alpha_rd": 0.1,  # random deletion
        "num_aug": 4
    },
    "llm": {
        "temperature": 0.7,
        "max_tokens": 150,
        "num_variations": 3
    }
}

# Evaluation settings
DEFAULT_EVALUATION = {
    "metrics": ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "accuracy"],
    "threshold_tuning": {
        "enabled": True,
        "metric": "f1_macro",
        "per_class": False,
        "n_thresholds": 100,
        "threshold_range": [0.1, 0.9]
    }
}


def get_default_config() -> Dict[str, Any]:
    """
    Get the default configuration dictionary.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "data": {
            "data_path": DEFAULT_DATA_PATH,
            "goemotions_files": GOEMOTIONS_FILES,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "random_seed": 42,
            "min_length": 10,
            "max_length": 512,
            "remove_duplicates": True,
            "balance_classes": False
        },
        "model": {
            "model_name": "bert-base-uncased",
            "num_labels": len(EMOTION_LABELS),
            "max_length": 512,
            "dropout_rate": 0.1,
            "hidden_size": None
        },
        "training": {
            **DEFAULT_HYPERPARAMETERS,
            "output_dir": DEFAULT_OUTPUT_PATH,
            "save_strategy": "steps",
            "save_steps": 1000,
            "eval_strategy": "steps",
            "eval_steps": 500,
            "logging_steps": 100,
            "early_stopping": True,
            "early_stopping_patience": 3,
            "fp16": False,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0
        },
        "evaluation": DEFAULT_EVALUATION,
        "augmentation": DEFAULT_AUGMENTATION,
        "preprocessing": {
            "lowercase": True,
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": False,
            "remove_extra_whitespace": True,
            "remove_non_ascii": False
        },
        "labels": {
            "emotion_labels": EMOTION_LABELS,
            "label_to_id": EMOTION_LABEL_TO_ID,
            "id_to_label": ID_TO_EMOTION_LABEL
        }
    }


def get_emotion_label_groups() -> Dict[str, List[str]]:
    """
    Get emotion label groups for analysis.
    
    Returns:
        Dictionary mapping group names to emotion labels
    """
    return {
        "positive": [
            "admiration", "amusement", "approval", "caring", "excitement", 
            "gratitude", "joy", "love", "optimism", "pride", "relief"
        ],
        "negative": [
            "anger", "annoyance", "disappointment", "disapproval", "disgust",
            "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"
        ],
        "neutral": ["neutral", "realization"],
        "ambiguous": [
            "confusion", "curiosity", "desire", "surprise"
        ]
    }


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model-specific configuration.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
    """
    base_config = {
        "num_labels": len(EMOTION_LABELS),
        "max_length": 512,
        "dropout_rate": 0.1
    }
    
    model_specific = {
        "bert-base-uncased": {
            "hidden_size": 768,
            "learning_rate": 2e-5,
            "batch_size": 16
        },
        "bert-base-cased": {
            "hidden_size": 768,
            "learning_rate": 2e-5,
            "batch_size": 16
        },
        "distilbert-base-uncased": {
            "hidden_size": 768,
            "learning_rate": 5e-5,
            "batch_size": 32
        },
        "roberta-base": {
            "hidden_size": 768,
            "learning_rate": 1e-5,
            "batch_size": 16
        },
        "albert-base-v2": {
            "hidden_size": 768,
            "learning_rate": 3e-5,
            "batch_size": 16
        }
    }
    
    config = base_config.copy()
    if model_name in model_specific:
        config.update(model_specific[model_name])
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ["data", "model", "training", "evaluation"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate data config
    data_config = config["data"]
    if "data_path" not in data_config:
        raise ValueError("Missing 'data_path' in data config")
    
    # Validate model config
    model_config = config["model"]
    if "model_name" not in model_config:
        raise ValueError("Missing 'model_name' in model config")
    
    if model_config["model_name"] not in SUPPORTED_MODELS:
        print(f"Warning: {model_config['model_name']} is not in supported models list")
    
    # Validate training config
    training_config = config["training"]
    required_training_keys = ["learning_rate", "batch_size", "num_epochs", "output_dir"]
    
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing '{key}' in training config")
    
    # Validate splits sum to 1
    if "train_split" in data_config and "val_split" in data_config and "test_split" in data_config:
        total_split = data_config["train_split"] + data_config["val_split"] + data_config["test_split"]
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    return True
