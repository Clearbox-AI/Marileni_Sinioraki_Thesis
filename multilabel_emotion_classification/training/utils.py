"""Training utilities."""

import os
import logging
from typing import Dict, Any, Optional
from .trainer import TrainingConfig, create_training_arguments

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "info"):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def prepare_output_directory(output_dir: str) -> str:
    """
    Prepare the output directory for training.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Prepared output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory prepared: {output_dir}")
    return output_dir


def save_training_config(config: TrainingConfig, output_dir: str):
    """
    Save training configuration to file.
    
    Args:
        config: Training configuration
        output_dir: Output directory
    """
    import json
    from dataclasses import asdict
    
    config_path = os.path.join(output_dir, "training_config.json")
    
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
        
    logger.info(f"Training configuration saved to: {config_path}")


def load_training_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration
    """
    import json
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
        
    return TrainingConfig(**config_dict)


def setup_training_environment(config: TrainingConfig):
    """
    Set up the training environment.
    
    Args:
        config: Training configuration
    """
    # Set up logging
    setup_logging(config.log_level)
    
    # Prepare output directory
    prepare_output_directory(config.output_dir)
    
    # Save configuration
    save_training_config(config, config.output_dir)
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    try:
        import torch
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    except ImportError:
        logger.warning("PyTorch not available, skipping torch seed setting")
    
    logger.info(f"Training environment set up with seed: {config.seed}")


def get_device():
    """
    Get the device for training.
    
    Returns:
        Device string
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        return device
    except ImportError:
        logger.warning("PyTorch not available, returning 'cpu'")
        return "cpu"


def print_training_summary(config: TrainingConfig):
    """
    Print a summary of training configuration.
    
    Args:
        config: Training configuration
    """
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Max length: {config.max_length}")
    print(f"Output directory: {config.output_dir}")
    print(f"Early stopping: {config.early_stopping}")
    print(f"FP16: {config.fp16}")
    print(f"Seed: {config.seed}")
    print("="*50 + "\n")
