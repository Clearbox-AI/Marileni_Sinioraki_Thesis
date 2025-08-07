"""Utility functions for multilabel emotion classification."""

import os
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    # Convert string to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Configure logging
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    import json
    
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary loaded from JSON
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def print_system_info():
    """Print system information."""
    import platform
    import sys
    
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Check for GPU
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch not available")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not available")
    
    print("="*50 + "\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def count_parameters(model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    try:
        import torch.nn as nn
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError:
        return 0


def get_device() -> str:
    """
    Get the best available device for computation.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    except ImportError:
        return "cpu"


def memory_usage() -> Dict[str, float]:
    """
    Get memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percentage": memory.percent
    }
    
    # GPU memory if available
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_memory_stats()
            memory_info["gpu_allocated_gb"] = gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3)
            memory_info["gpu_reserved_gb"] = gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3)
    except ImportError:
        pass
    
    return memory_info


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create a unique experiment directory.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Path to the created experiment directory
    """
    timestamp = get_timestamp()
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    ensure_dir(experiment_dir)
    return experiment_dir


def save_experiment_info(
    experiment_dir: str,
    config: Dict[str, Any],
    model_info: Optional[Dict[str, Any]] = None,
    system_info: bool = True
):
    """
    Save experiment information to the experiment directory.
    
    Args:
        experiment_dir: Experiment directory
        config: Configuration dictionary
        model_info: Model information
        system_info: Whether to include system information
    """
    info = {
        "timestamp": get_timestamp(),
        "config": config,
        "model_info": model_info or {},
        "system_info": {}
    }
    
    if system_info:
        info["system_info"] = {
            "platform": __import__('platform').platform(),
            "python_version": __import__('sys').version,
        }
        
        try:
            import torch
            info["system_info"]["pytorch_version"] = torch.__version__
            info["system_info"]["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["system_info"]["cuda_version"] = torch.version.cuda
                info["system_info"]["gpu_count"] = torch.cuda.device_count()
        except ImportError:
            pass
    
    save_json(info, os.path.join(experiment_dir, "experiment_info.json"))


def load_experiment_info(experiment_dir: str) -> Dict[str, Any]:
    """
    Load experiment information from directory.
    
    Args:
        experiment_dir: Experiment directory
        
    Returns:
        Experiment information dictionary
    """
    return load_json(os.path.join(experiment_dir, "experiment_info.json"))


def compare_experiments(experiment_dirs: List[str]) -> pd.DataFrame:
    """
    Compare multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directories
        
    Returns:
        DataFrame with experiment comparison
    """
    comparison_data = []
    
    for exp_dir in experiment_dirs:
        try:
            info = load_experiment_info(exp_dir)
            
            # Extract key metrics
            config = info.get("config", {})
            model_info = info.get("model_info", {})
            
            row = {
                "experiment": os.path.basename(exp_dir),
                "timestamp": info.get("timestamp", "unknown"),
                "model_name": config.get("model", {}).get("model_name", "unknown"),
                "learning_rate": config.get("training", {}).get("learning_rate", "unknown"),
                "batch_size": config.get("training", {}).get("batch_size", "unknown"),
                "num_epochs": config.get("training", {}).get("num_epochs", "unknown"),
                "f1_macro": model_info.get("f1_macro", "unknown"),
                "f1_micro": model_info.get("f1_micro", "unknown"),
                "accuracy": model_info.get("accuracy", "unknown"),
            }
            
            comparison_data.append(row)
            
        except Exception as e:
            print(f"Error loading experiment {exp_dir}: {e}")
    
    return pd.DataFrame(comparison_data)


def validate_data_splits(
    train_split: float, 
    val_split: float, 
    test_split: float,
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that data splits sum to 1.0.
    
    Args:
        train_split: Training split
        val_split: Validation split
        test_split: Test split
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if valid, False otherwise
    """
    total = train_split + val_split + test_split
    return abs(total - 1.0) <= tolerance


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten a nested list.
    
    Args:
        nested_list: Nested list to flatten
        
    Returns:
        Flattened list
    """
    return [item for sublist in nested_list for item in sublist]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    return numerator / denominator if denominator != 0 else default


def print_progress_bar(
    iteration: int, 
    total: int, 
    prefix: str = '', 
    suffix: str = '', 
    length: int = 50
):
    """
    Print a progress bar.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Length of progress bar
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    
    if iteration == total:
        print()  # New line when complete
