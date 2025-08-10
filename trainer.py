"""
Training pipeline for GoEmotions multi-label emotion classification.
"""

import os
import numpy as np
from typing import Dict, Tuple
from transformers import TrainingArguments
from model import MultiLabelBERT, MultilabelTrainer
from data_loader import GoEmotionsDataLoader


class GoEmotionsTrainingPipeline:
    """
    Complete training pipeline for GoEmotions multi-label emotion classification.
    """
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 output_dir: str = "./results",
                 num_labels: int = 27):
        """
        Initialize training pipeline.
        
        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save results
            num_labels: Number of emotion labels
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_labels = num_labels
        
        # Initialize components
        self.data_loader = GoEmotionsDataLoader()
        self.model_wrapper = MultiLabelBERT(model_name, num_labels)
        self.trainer_wrapper = MultilabelTrainer()
        
        # Training state
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.trainer = None
        self.thresholds = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def prepare_data(self, 
                    tokenizer_name: str = None,
                    max_length: int = 128,
                    train_ratio: float = 0.6,
                    val_ratio: float = 0.2,
                    test_ratio: float = 0.2):
        """
        Prepare and tokenize datasets for training.
        
        Args:
            tokenizer_name: Tokenizer to use (defaults to model_name)
            max_length: Maximum sequence length
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        if tokenizer_name is None:
            tokenizer_name = self.model_name
            
        print("Preparing datasets for training...")
        
        # Prepare tokenized datasets
        self.train_ds, self.val_ds, self.test_ds = self.data_loader.prepare_training_datasets(
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        print("Data preparation complete!")
        
    def setup_training_args(self,
                           learning_rate: float = 2e-5,
                           train_batch_size: int = 16,
                           eval_batch_size: int = 32,
                           num_epochs: int = 8,
                           weight_decay: float = 0.01,
                           warmup_steps: int = 500,
                           eval_strategy: str = "epoch",
                           save_strategy: str = "epoch",
                           logging_strategy: str = "epoch",
                           load_best_model_at_end: bool = True,
                           metric_for_best_model: str = "eval_micro/f1",
                           greater_is_better: bool = True,
                           save_total_limit: int = 2,
                           report_to: str = "wandb",
                           run_name: str = "goemotions-multilabel-bert") -> TrainingArguments:
        """
        Setup training arguments with optimized defaults for multi-label classification.
        
        Returns:
            TrainingArguments object
        """
        return TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            logging_strategy=logging_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
            num_train_epochs=num_epochs,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_dir="./logs",
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            save_total_limit=save_total_limit,
            report_to=report_to,
            run_name=run_name,
            push_to_hub=False,
            dataloader_num_workers=2
        )
    
    def create_trainer(self, training_args: TrainingArguments):
        """Create the trainer with custom loss function."""
        
        def compute_metrics_wrapper(eval_pred):
            return self.model_wrapper.compute_metrics(eval_pred, self.thresholds)
        
        self.trainer = self.trainer_wrapper.create_trainer(
            model=self.model_wrapper.model,
            args=training_args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            tokenizer=self.model_wrapper.tokenizer,
            compute_metrics_fn=compute_metrics_wrapper
        )
        
    def train(self, training_args: TrainingArguments = None):
        """
        Execute the complete training pipeline.
        
        Args:
            training_args: Training arguments (uses defaults if None)
        """
        if self.train_ds is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        if training_args is None:
            training_args = self.setup_training_args()
        
        # Create trainer
        self.create_trainer(training_args)
        
        print("Starting training...")
        print(f"Training samples: {len(self.train_ds)}")
        print(f"Validation samples: {len(self.val_ds)}")
        print(f"Test samples: {len(self.test_ds)}")
        
        # Train the model
        train_result = self.trainer.train()
        
        print("Training completed!")
        return train_result
    
    def evaluate_and_tune_thresholds(self) -> Tuple[Dict, np.ndarray]:
        """
        Evaluate on validation set and tune thresholds for optimal performance.
        
        Returns:
            Tuple of (metrics, thresholds)
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Evaluating on validation set and tuning thresholds...")
        
        # Get predictions on validation set
        eval_output = self.trainer.predict(self.val_ds)
        val_logits = eval_output.predictions
        val_labels = eval_output.label_ids
        
        # Tune thresholds
        self.thresholds = self.model_wrapper.tune_thresholds(val_logits, val_labels)
        
        # Compute metrics with tuned thresholds
        metrics = self.model_wrapper.compute_metrics((val_logits, val_labels), self.thresholds)
        
        print("Validation metrics with tuned thresholds:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Print detailed per-class report
        self.model_wrapper.print_class_report(val_logits, val_labels, self.thresholds)
        
        return metrics, self.thresholds
    
    def evaluate_test_set(self) -> Dict:
        """
        Evaluate on test set using tuned thresholds.
        
        Returns:
            Test set metrics
        """
        if self.thresholds is None:
            raise ValueError("Thresholds not tuned. Call evaluate_and_tune_thresholds() first.")
        
        print("Evaluating on test set...")
        
        # Get predictions on test set
        test_output = self.trainer.predict(self.test_ds)
        test_logits = test_output.predictions
        test_labels = test_output.label_ids
        
        # Compute metrics with tuned thresholds
        test_metrics = self.model_wrapper.compute_metrics((test_logits, test_labels), self.thresholds)
        
        print("Test set metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Print detailed per-class report
        test_report = self.model_wrapper.print_class_report(test_logits, test_labels, self.thresholds)
        
        return test_metrics
    
    def evaluate_with_default_thresholds(self) -> Dict:
        """
        Evaluate on test set using default 0.5 thresholds (for comparison).
        
        Returns:
            Test set metrics with default thresholds
        """
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Evaluating with default 0.5 thresholds...")
        
        # Temporarily set compute_metrics without thresholds
        def compute_metrics_default(eval_pred):
            return self.model_wrapper.compute_metrics(eval_pred, thresholds=None)
        
        # Update trainer's compute_metrics
        original_compute_metrics = self.trainer.compute_metrics
        self.trainer.compute_metrics = compute_metrics_default
        
        # Evaluate
        metrics = self.trainer.evaluate(self.test_ds)
        
        # Restore original compute_metrics
        self.trainer.compute_metrics = original_compute_metrics
        
        print("Metrics with default thresholds:")
        for key, value in metrics.items():
            if 'eval_' in key:
                print(f"{key}: {value:.4f}")
        
        return metrics
    
    def run_complete_pipeline(self, 
                            training_args: TrainingArguments = None,
                            **data_kwargs) -> Dict:
        """
        Run the complete training and evaluation pipeline.
        
        Args:
            training_args: Training arguments
            **data_kwargs: Arguments for data preparation
            
        Returns:
            Dictionary with all results
        """
        print("=" * 60)
        print("GOEMOTIONS MULTI-LABEL EMOTION CLASSIFICATION PIPELINE")
        print("=" * 60)
        
        # 1. Prepare data
        self.prepare_data(**data_kwargs)
        
        # 2. Train model
        train_result = self.train(training_args)
        
        # 3. Tune thresholds and evaluate validation set
        val_metrics, thresholds = self.evaluate_and_tune_thresholds()
        
        # 4. Evaluate test set with tuned thresholds
        test_metrics = self.evaluate_test_set()
        
        # 5. Compare with default thresholds
        default_metrics = self.evaluate_with_default_thresholds()
        
        results = {
            'train_result': train_result,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'default_metrics': default_metrics,
            'thresholds': thresholds
        }
        
        print("=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        
        return results