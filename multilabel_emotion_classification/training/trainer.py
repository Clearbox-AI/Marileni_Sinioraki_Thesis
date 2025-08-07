"""Training utilities and trainer class for multilabel emotion classification."""

import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model and data
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    
    # Training parameters
    learning_rate: float = 2e-5
    num_epochs: int = 3
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
    
    # Evaluation
    eval_steps: int = 500
    eval_strategy: str = "steps"  # "steps" or "epoch"
    save_steps: int = 1000
    save_strategy: str = "steps"  # "steps" or "epoch"
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
    
    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"emotion_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class EmotionTrainer(Trainer):
    """Custom trainer for multilabel emotion classification."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = float('-inf')
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss for multilabel classification.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss and optionally outputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.get("logits")
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval"
    ):
        """
        Run evaluation and return metrics.
        
        Args:
            eval_dataset: Dataset to evaluate on
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        self.log(output.metrics)
        
        # Check if this is the best model so far
        if self.compute_metrics is not None:
            # Assume we're tracking F1 score as the main metric
            current_metric = output.metrics.get(f"{metric_key_prefix}_f1_macro", 0)
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                logger.info(f"New best model with {metric_key_prefix}_f1_macro: {current_metric:.4f}")
        
        return output.metrics
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step.
        
        Args:
            model: The model
            inputs: Input batch
            prediction_loss_only: Whether to return only loss
            ignore_keys: Keys to ignore
            
        Returns:
            Prediction step results
        """
        inputs = self._prepare_inputs(inputs)
        
        if ignore_keys is None:
            ignore_keys = []
            
        with torch.no_grad():
            labels = inputs.pop("labels") if "labels" in inputs else None
            
            outputs = model(**inputs)
            
            if labels is not None:
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    logits = outputs.get("logits")
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels.float())
            else:
                loss = None
                
            logits = outputs.get("logits")
            
        if prediction_loss_only:
            return (loss, None, None)
            
        # Convert logits to predictions
        predictions = torch.sigmoid(logits) if logits is not None else None
        
        return (loss, predictions, labels)


def create_training_arguments(config: TrainingConfig) -> TrainingArguments:
    """
    Create TrainingArguments from config.
    
    Args:
        config: Training configuration
        
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        adam_epsilon=config.adam_epsilon,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy=config.eval_strategy,
        save_steps=config.save_steps,
        save_strategy=config.save_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=config.seed,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        run_name=config.run_name,
        report_to=None,  # Disable wandb/tensorboard by default
    )


def setup_training(
    model,
    train_dataset,
    eval_dataset,
    config: TrainingConfig,
    compute_metrics=None,
    tokenizer=None
) -> EmotionTrainer:
    """
    Set up the trainer for training.
    
    Args:
        model: The model to train
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Training configuration
        compute_metrics: Function to compute metrics
        tokenizer: Tokenizer (optional)
        
    Returns:
        Configured trainer
    """
    # Create training arguments
    training_args = create_training_arguments(config)
    
    # Set up callbacks
    callbacks = []
    if config.early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=config.early_stopping_threshold
            )
        )
    
    # Create trainer
    trainer = EmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    return trainer


def train_model(
    trainer: EmotionTrainer,
    resume_from_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        trainer: Configured trainer
        resume_from_checkpoint: Path to checkpoint to resume from
        
    Returns:
        Training results
    """
    logger.info("Starting training...")
    
    # Train the model
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the model
    trainer.save_model()
    
    # Log training results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Training completed!")
    
    return train_result


def evaluate_model(trainer: EmotionTrainer) -> Dict[str, float]:
    """
    Evaluate the trained model.
    
    Args:
        trainer: Trained trainer
        
    Returns:
        Evaluation metrics
    """
    logger.info("Starting evaluation...")
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    
    # Log evaluation results
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    
    logger.info("Evaluation completed!")
    
    return eval_result
