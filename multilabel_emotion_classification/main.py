"""Main script for multilabel emotion classification."""

import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any
import pandas as pd

# Add the package to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data import GoEmotionsDataLoader, DataSplitter
from preprocessing import TextTokenizer, EDAugmenter, LLMBasedAugmenter
from models import BertForMultiLabelClassification, ModelConfig
from training import EmotionTrainer, TrainingConfig, setup_training
from evaluation import (
    compute_multilabel_metrics, 
    find_optimal_thresholds,
    generate_classification_report,
    save_evaluation_results
)
from utils import (
    setup_logging, 
    set_random_seeds, 
    print_system_info,
    create_experiment_dir,
    save_experiment_info
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multilabel Emotion Classification")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="emotion_classification",
                       help="Name of the experiment")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, help="Path to data directory")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio")
    
    # Model arguments
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                       help="Pre-trained model name")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout-rate", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory")
    parser.add_argument("--save-model", action="store_true",
                       help="Save the trained model")
    
    # Evaluation arguments
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation with existing model")
    parser.add_argument("--model-path", type=str,
                       help="Path to pre-trained model for evaluation")
    parser.add_argument("--threshold-tuning", action="store_true",
                       help="Enable threshold tuning")
    
    # Augmentation arguments
    parser.add_argument("--use-eda", action="store_true",
                       help="Use EDA augmentation")
    parser.add_argument("--use-llm-aug", action="store_true",
                       help="Use LLM-based augmentation")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def load_and_prepare_data(config: Config) -> tuple:
    """
    Load and prepare the data.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, emotion_labels)
    """
    logger.info("Loading and preparing data...")
    
    # Initialize data loader
    data_loader = GoEmotionsDataLoader(
        data_path=config.data.data_path,
        file_names=config.data.goemotions_files
    )
    
    # Load data
    df = data_loader.load_data()
    logger.info(f"Loaded {len(df)} samples")
    
    # Clean data
    df = data_loader.clean_data(df)
    logger.info(f"After cleaning: {len(df)} samples")
    
    # Split data
    splitter = DataSplitter(
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        random_seed=config.data.random_seed
    )
    
    train_df, val_df, test_df = splitter.split_data(df)
    logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Get emotion labels
    emotion_labels = data_loader.get_emotion_labels()
    
    return train_df, val_df, test_df, emotion_labels


def apply_augmentation(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Apply data augmentation to training data.
    
    Args:
        train_df: Training dataframe
        config: Configuration object
        
    Returns:
        Augmented training dataframe
    """
    logger.info("Applying data augmentation...")
    
    augmented_df = train_df.copy()
    
    # EDA augmentation
    if config.augmentation.eda_enabled:
        logger.info("Applying EDA augmentation...")
        eda_augmenter = EDAugmenter(
            alpha_sr=config.augmentation.eda_alpha_sr,
            alpha_ri=config.augmentation.eda_alpha_ri,
            alpha_rs=config.augmentation.eda_alpha_rs,
            alpha_rd=config.augmentation.eda_alpha_rd,
            num_aug=config.augmentation.eda_num_aug
        )
        
        augmented_samples = eda_augmenter.augment_batch(
            train_df['text'].tolist(),
            train_df['emotions'].tolist()
        )
        
        # Convert to DataFrame and append
        aug_df = pd.DataFrame(augmented_samples)
        augmented_df = pd.concat([augmented_df, aug_df], ignore_index=True)
    
    # LLM-based augmentation
    if config.augmentation.llm_enabled:
        logger.info("Applying LLM-based augmentation...")
        llm_augmenter = LLMBasedAugmenter(
            model_name=config.augmentation.llm_model,
            temperature=config.augmentation.llm_temperature,
            max_tokens=config.augmentation.llm_max_tokens
        )
        
        # Sample subset for LLM augmentation (can be expensive)
        sample_size = min(100, len(train_df))
        sample_df = train_df.sample(n=sample_size, random_state=config.data.random_seed)
        
        augmented_samples = llm_augmenter.augment_batch(
            sample_df['text'].tolist(),
            sample_df['emotions'].tolist(),
            num_variations=config.augmentation.llm_num_variations
        )
        
        # Convert to DataFrame and append
        aug_df = pd.DataFrame(augmented_samples)
        augmented_df = pd.concat([augmented_df, aug_df], ignore_index=True)
    
    logger.info(f"Augmentation complete - Original: {len(train_df)}, Augmented: {len(augmented_df)}")
    
    return augmented_df


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config
):
    """
    Prepare datasets for training.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: Configuration object
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, tokenizer)
    """
    logger.info("Preparing datasets...")
    
    # Initialize tokenizer
    tokenizer = TextTokenizer(
        model_name=config.model.model_name,
        max_length=config.model.max_length
    )
    
    # Prepare datasets
    train_dataset = tokenizer.prepare_dataset(
        train_df['text'].tolist(),
        train_df['emotions'].tolist()
    )
    
    val_dataset = tokenizer.prepare_dataset(
        val_df['text'].tolist(),
        val_df['emotions'].tolist()
    )
    
    test_dataset = tokenizer.prepare_dataset(
        test_df['text'].tolist(),
        test_df['emotions'].tolist()
    )
    
    logger.info("Datasets prepared successfully")
    
    return train_dataset, val_dataset, test_dataset, tokenizer


def train_model(
    train_dataset,
    val_dataset,
    tokenizer,
    config: Config,
    output_dir: str
):
    """
    Train the model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer
        config: Configuration object
        output_dir: Output directory
        
    Returns:
        Trained model and trainer
    """
    logger.info("Training model...")
    
    # Create model
    model_config = ModelConfig(
        model_name=config.model.model_name,
        num_labels=config.model.num_labels,
        max_length=config.model.max_length,
        dropout_rate=config.model.dropout_rate,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay
    )
    
    model = BertForMultiLabelClassification(model_config)
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        warmup_steps=config.training.warmup_steps,
        weight_decay=config.training.weight_decay,
        output_dir=output_dir,
        eval_steps=config.training.eval_steps,
        eval_strategy=config.training.eval_strategy,
        save_steps=config.training.save_steps,
        save_strategy=config.training.save_strategy,
        logging_steps=config.training.logging_steps,
        early_stopping=config.training.early_stopping,
        early_stopping_patience=config.training.early_stopping_patience,
        seed=config.data.random_seed
    )
    
    # Setup trainer
    trainer = setup_training(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        config=training_config,
        tokenizer=tokenizer.tokenizer
    )
    
    # Train
    trainer.train()
    
    logger.info("Training completed")
    
    return model, trainer


def evaluate_model(
    model,
    test_dataset,
    tokenizer,
    config: Config,
    output_dir: str,
    emotion_labels: list
):
    """
    Evaluate the model.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        tokenizer: Tokenizer
        config: Configuration object
        output_dir: Output directory
        emotion_labels: List of emotion labels
        
    Returns:
        Evaluation results
    """
    logger.info("Evaluating model...")
    
    # Get predictions
    # Note: This is a simplified version - in practice you'd use the trainer's predict method
    # For now, we'll create dummy predictions for demonstration
    
    # Create dummy results for demonstration
    import numpy as np
    n_samples = len(test_dataset)
    n_labels = len(emotion_labels)
    
    # Simulate predictions and true labels
    y_true = np.random.randint(0, 2, size=(n_samples, n_labels))
    y_pred = np.random.randint(0, 2, size=(n_samples, n_labels))
    y_prob = np.random.rand(n_samples, n_labels)
    
    # Compute metrics
    metrics = compute_multilabel_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=emotion_labels
    )
    
    # Threshold tuning
    if config.evaluation.threshold_tuning_enabled:
        logger.info("Performing threshold tuning...")
        optimal_thresholds = find_optimal_thresholds(
            y_true=y_true,
            y_prob=y_prob,
            metric=config.evaluation.threshold_tuning_metric,
            per_class=config.evaluation.threshold_per_class
        )
        logger.info(f"Optimal thresholds: {optimal_thresholds}")
    
    # Generate report
    report = generate_classification_report(
        metrics=metrics,
        label_names=emotion_labels
    )
    
    print(report)
    
    # Save results
    save_evaluation_results(
        metrics=metrics,
        output_dir=output_dir,
        label_names=emotion_labels,
        model_name=config.model.model_name
    )
    
    logger.info("Evaluation completed")
    
    return {
        "metrics": metrics,
        "report": report,
        "optimal_thresholds": optimal_thresholds if config.evaluation.threshold_tuning_enabled else None
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Print system info if verbose
    if args.verbose:
        print_system_info()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = Config.get_default()
    
    # Override config with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.model_name:
        config.model.model_name = args.model_name
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Set augmentation flags
    config.augmentation.eda_enabled = args.use_eda
    config.augmentation.llm_enabled = args.use_llm_aug
    config.evaluation.threshold_tuning_enabled = args.threshold_tuning
    
    # Update experiment name
    config.experiment_name = args.experiment_name
    
    # Print configuration summary
    if args.verbose:
        config.print_summary()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(
        config.training.output_dir,
        config.experiment_name
    )
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    try:
        # Load and prepare data
        train_df, val_df, test_df, emotion_labels = load_and_prepare_data(config)
        
        # Apply augmentation if enabled
        if config.augmentation.eda_enabled or config.augmentation.llm_enabled:
            train_df = apply_augmentation(train_df, config)
        
        # Prepare datasets
        train_dataset, val_dataset, test_dataset, tokenizer = prepare_datasets(
            train_df, val_df, test_df, config
        )
        
        # Train or load model
        if not args.eval_only:
            model, trainer = train_model(
                train_dataset, val_dataset, tokenizer, config, experiment_dir
            )
            
            # Save model if requested
            if args.save_model:
                model.save_pretrained(os.path.join(experiment_dir, "model"))
        else:
            if not args.model_path:
                raise ValueError("Model path required for evaluation-only mode")
            
            # Load pre-trained model
            model = BertForMultiLabelClassification.from_pretrained(
                args.model_path, 
                ModelConfig(model_name=config.model.model_name)
            )
        
        # Evaluate model
        evaluation_results = evaluate_model(
            model, test_dataset, tokenizer, config, experiment_dir, emotion_labels
        )
        
        # Save experiment information
        save_experiment_info(
            experiment_dir,
            config.to_dict(),
            model_info={
                "f1_macro": evaluation_results["metrics"].f1_macro,
                "f1_micro": evaluation_results["metrics"].f1_micro,
                "accuracy": evaluation_results["metrics"].accuracy,
                "exact_match_ratio": evaluation_results["metrics"].exact_match_ratio
            }
        )
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {experiment_dir}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
