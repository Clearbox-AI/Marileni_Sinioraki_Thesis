"""Configuration classes for multilabel emotion classification."""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from .settings import get_default_config, validate_config, EMOTION_LABELS


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str = "./data"
    goemotions_files: List[str] = field(default_factory=lambda: [
        "goemotions_1.csv", "goemotions_2.csv", "goemotions_3.csv"
    ])
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    min_length: int = 10
    max_length: int = 512
    remove_duplicates: bool = True
    balance_classes: bool = False
    
    def __post_init__(self):
        # Validate splits sum to 1
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


@dataclass 
class ModelConfig:
    """Model configuration."""
    model_name: str = "bert-base-uncased"
    num_labels: int = len(EMOTION_LABELS)
    max_length: int = 512
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_length: int = 512
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Scheduler
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Evaluation and saving
    eval_steps: int = 500
    eval_strategy: str = "steps"
    save_steps: int = 1000
    save_strategy: str = "steps"
    save_total_limit: int = 3
    
    # Logging
    logging_steps: int = 100
    log_level: str = "info"
    
    # Output
    output_dir: str = "./results"
    run_name: Optional[str] = None
    seed: int = 42
    
    # Hardware
    fp16: bool = False
    dataloader_num_workers: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "f1_macro", "f1_micro", "precision_macro", "recall_macro", "accuracy"
    ])
    threshold_tuning_enabled: bool = True
    threshold_tuning_metric: str = "f1_macro"
    threshold_per_class: bool = False
    n_thresholds: int = 100
    threshold_range: List[float] = field(default_factory=lambda: [0.1, 0.9])
    save_predictions: bool = True
    save_probabilities: bool = True
    create_plots: bool = True


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    # EDA (Easy Data Augmentation)
    eda_enabled: bool = False
    eda_alpha_sr: float = 0.1  # synonym replacement
    eda_alpha_ri: float = 0.1  # random insertion
    eda_alpha_rs: float = 0.1  # random swap
    eda_alpha_rd: float = 0.1  # random deletion
    eda_num_aug: int = 4
    
    # LLM-based augmentation
    llm_enabled: bool = False
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 150
    llm_num_variations: int = 3
    
    # Synthetic data generation
    synthetic_enabled: bool = False
    synthetic_samples_per_class: int = 100
    
    # Downsampling
    downsample_enabled: bool = False
    downsample_strategy: str = "random"  # "random" or "cluster"
    downsample_ratio: float = 0.5


@dataclass
class PreprocessingConfig:
    """Text preprocessing configuration."""
    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    remove_hashtags: bool = False
    remove_extra_whitespace: bool = True
    remove_non_ascii: bool = False
    min_length: int = 10
    max_length: int = 512


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    
    # Metadata
    experiment_name: str = "emotion_classification"
    description: str = "Multilabel emotion classification with BERT"
    version: str = "1.0.0"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create Config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Config object
        """
        # Validate config
        validate_config(config_dict)
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            augmentation=AugmentationConfig(**config_dict.get("augmentation", {})),
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            experiment_name=config_dict.get("experiment_name", "emotion_classification"),
            description=config_dict.get("description", "Multilabel emotion classification with BERT"),
            version=config_dict.get("version", "1.0.0")
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """
        Load config from JSON file.
        
        Args:
            json_path: Path to JSON config file
            
        Returns:
            Config object
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def get_default(cls) -> 'Config':
        """
        Get default configuration.
        
        Returns:
            Default Config object
        """
        default_dict = get_default_config()
        return cls.from_dict(default_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return asdict(self)
    
    def to_json(self, json_path: str):
        """
        Save config to JSON file.
        
        Args:
            json_path: Path to save JSON config file
        """
        config_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update config from dictionary.
        
        Args:
            updates: Dictionary with updates
        """
        for section, section_updates in updates.items():
            if hasattr(self, section) and isinstance(section_updates, dict):
                config_section = getattr(self, section)
                for key, value in section_updates.items():
                    if hasattr(config_section, key):
                        setattr(config_section, key, value)
            elif hasattr(self, section):
                setattr(self, section, section_updates)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        config_dict = self.to_dict()
        return validate_config(config_dict)
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        print(f"Experiment: {self.experiment_name}")
        print(f"Description: {self.description}")
        print(f"Version: {self.version}")
        print()
        
        print("Data Configuration:")
        print(f"  Data path: {self.data.data_path}")
        print(f"  Train/Val/Test split: {self.data.train_split}/{self.data.val_split}/{self.data.test_split}")
        print(f"  Random seed: {self.data.random_seed}")
        print()
        
        print("Model Configuration:")
        print(f"  Model name: {self.model.model_name}")
        print(f"  Number of labels: {self.model.num_labels}")
        print(f"  Max length: {self.model.max_length}")
        print(f"  Dropout rate: {self.model.dropout_rate}")
        print()
        
        print("Training Configuration:")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Number of epochs: {self.training.num_epochs}")
        print(f"  Output directory: {self.training.output_dir}")
        print(f"  Early stopping: {self.training.early_stopping}")
        print()
        
        print("Evaluation Configuration:")
        print(f"  Metrics: {', '.join(self.evaluation.metrics)}")
        print(f"  Threshold tuning: {self.evaluation.threshold_tuning_enabled}")
        print()
        
        print("Augmentation Configuration:")
        print(f"  EDA enabled: {self.augmentation.eda_enabled}")
        print(f"  LLM enabled: {self.augmentation.llm_enabled}")
        print(f"  Synthetic enabled: {self.augmentation.synthetic_enabled}")
        print("="*60 + "\n")
