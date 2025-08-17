"""
Training pipeline for GoEmotions multi-label emotion classification.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
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
    
    def run_downsampling_experiments(self, 
                                   reduction_percentages: List[float] = None,
                                   random_state: int = 123,
                                   **data_kwargs) -> pd.DataFrame:
        """
        Run downsampling experiments across multiple reduction levels.
        
        Args:
            reduction_percentages: List of reduction percentages to test
            random_state: Random seed for reproducibility
            **data_kwargs: Arguments for data preparation
            
        Returns:
            DataFrame with experiment results
        """
        if reduction_percentages is None:
            reduction_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        # Prepare initial data
        print("Preparing data for downsampling experiments...")
        self.prepare_data(**data_kwargs)
        
        # Get raw DataFrames for downsampling
        train_df, val_df, test_df = self.data_loader.get_raw_dataframes(
            data_kwargs.get('train_ratio', 0.6),
            data_kwargs.get('val_ratio', 0.2), 
            data_kwargs.get('test_ratio', 0.2)
        )
        
        print(f"\nRunning downsampling experiments with reduction levels: {reduction_percentages}")
        print(f"Original training samples: {len(train_df)}")
        
        metrics_list = []
        
        for pct in reduction_percentages:
            print(f"\n{'='*60}")
            print(f"REDUCTION EXPERIMENT: {pct}%")
            print('='*60)
            
            try:
                # 1. Downsample training data
                down_train = self.data_loader.downsample_by_label_reduction(
                    train_df, reduction_pct=pct, random_state=random_state
                )
                
                # 2. Convert to HF datasets and tokenize
                train_ds = self.data_loader.convert_to_hf_dataset(down_train)
                val_ds = self.data_loader.convert_to_hf_dataset(val_df)
                test_ds = self.data_loader.convert_to_hf_dataset(test_df)
                
                tokenizer_name = data_kwargs.get('tokenizer_name', self.model_name)
                max_length = data_kwargs.get('max_length', 128)
                
                train_ds = self.data_loader.tokenize_dataset(train_ds, tokenizer_name, max_length)
                val_ds = self.data_loader.tokenize_dataset(val_ds, tokenizer_name, max_length)
                test_ds = self.data_loader.tokenize_dataset(test_ds, tokenizer_name, max_length)
                
                # 3. Create fresh model for this experiment
                self.model_wrapper = MultiLabelBERT(self.model_name, self.num_labels)
                
                # 4. Setup training arguments
                training_args = self.setup_training_args(
                    run_name=f"goemotions-downsample-{pct}pct"
                )
                
                # 5. Update datasets for this experiment
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.test_ds = test_ds
                
                # 6. Train model
                print(f"\nTraining on {len(down_train)} downsampled examples...")
                self.create_trainer(training_args)
                self.trainer.train()
                
                # 7. Evaluate and collect metrics
                print("Evaluating model...")
                val_metrics, thresholds = self.evaluate_and_tune_thresholds()
                test_metrics = self.evaluate_test_set()
                
                # 8. Store results
                result = {
                    'reduction_pct': pct,
                    'train_samples': len(down_train),
                    **{f"test_{k}": v for k, v in test_metrics.items()}
                }
                metrics_list.append(result)
                
                print(f"Completed {pct}% reduction experiment")
                
            except Exception as e:
                print(f"ERROR in {pct}% experiment: {e}")
                continue
        
        # Create results DataFrame
        if metrics_list:
            import matplotlib.pyplot as plt
            
            metrics_df = pd.DataFrame(metrics_list).set_index('reduction_pct')
            
            print(f"\n{'='*80}")
            print("DOWNSAMPLING EXPERIMENT SUMMARY")
            print('='*80)
            print(metrics_df[['train_samples', 'test_micro/f1', 'test_macro/f1']].round(4))
            
            # Plot results (your original plotting code)
            plt.figure(figsize=(8, 5))
            plt.plot(metrics_df.index, metrics_df["test_micro/f1"], marker="o", label="micro/F1")
            plt.plot(metrics_df.index, metrics_df["test_macro/f1"], marker="s", label="macro/F1")
            plt.xlabel("Downsampling Reduction (%)")
            plt.ylabel("F1 Score")
            plt.title("Test F1 vs. Training Set Downsampling")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            return metrics_df
        else:
            print("No successful experiments completed!")
            return pd.DataFrame()
    
    def analyze_underperforming_labels_after_downsampling(self,
                                                         reduction_pct: float = 50.0,
                                                         random_state: int = 123,
                                                         **data_kwargs) -> List[str]:
        """
        Run downsampling experiment and identify underperforming labels using regression analysis.
        
        Args:
            reduction_pct: Percentage reduction for downsampling
            random_state: Random seed
            **data_kwargs: Arguments for data preparation
            
        Returns:
            List of underperforming label names
        """
        print("="*80)
        print("DOWNSAMPLING + REGRESSION ANALYSIS EXPERIMENT")
        print("="*80)
        print(f"Downsample by {reduction_pct}% and identify underperforming labels")
        print("="*80)
        
        # Prepare initial data
        self.prepare_data(**data_kwargs)
        train_df, val_df, test_df = self.data_loader.get_raw_dataframes(
            data_kwargs.get('train_ratio', 0.6),
            data_kwargs.get('val_ratio', 0.2), 
            data_kwargs.get('test_ratio', 0.2)
        )
        
        # Downsample training data
        print(f"\n{'='*60}")
        print("DOWNSAMPLING EXPERIMENT")
        print('='*60)
        
        downsampled_train = self.data_loader.downsample_by_label_reduction(
            train_df, reduction_pct=reduction_pct, random_state=random_state
        )
        
        # Convert to HF datasets and tokenize
        train_ds = self.data_loader.convert_to_hf_dataset(downsampled_train)
        val_ds = self.data_loader.convert_to_hf_dataset(val_df)
        test_ds = self.data_loader.convert_to_hf_dataset(test_df)
        
        tokenizer_name = data_kwargs.get('tokenizer_name', self.model_name)
        max_length = data_kwargs.get('max_length', 128)
        
        train_ds = self.data_loader.tokenize_dataset(train_ds, tokenizer_name, max_length)
        val_ds = self.data_loader.tokenize_dataset(val_ds, tokenizer_name, max_length)
        test_ds = self.data_loader.tokenize_dataset(test_ds, tokenizer_name, max_length)
        
        # Create fresh model
        self.model_wrapper = MultiLabelBERT(self.model_name, self.num_labels)
        
        # Setup training
        training_args = self.setup_training_args(
            run_name=f"goemotions-analyze-{reduction_pct}pct"
        )
        
        # Update datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        
        # Train
        print(f"\nTraining model on {len(downsampled_train)} downsampled samples...")
        self.create_trainer(training_args)
        self.trainer.train()
        
        # Evaluate and tune thresholds
        print("\nEvaluating model...")
        val_metrics, thresholds = self.evaluate_and_tune_thresholds()
        test_metrics = self.evaluate_test_set()
        
        # Get test predictions for regression analysis
        test_output = self.trainer.predict(test_ds)
        
        # Identify underperforming labels using regression analysis
        print(f"\n{'='*60}")
        print("REGRESSION ANALYSIS FOR UNDERPERFORMING LABELS")
        print('='*60)
        
        underperforming_labels = self.model_wrapper.identify_underperforming_labels(
            test_output.predictions, 
            test_output.label_ids, 
            thresholds
        )
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print('='*80)
        print(f"Underperforming labels identified: {underperforming_labels}")
        print(f"Overall Test Micro F1: {test_metrics['micro/f1']:.4f}")
        print(f"Overall Test Macro F1: {test_metrics['macro/f1']:.4f}")
        print('='*80)
        
        return underperforming_labels
    
    def run_augmentation_experiments(self,
                                   underperforming_labels: List[str],
                                   reduction_pct: float = 60.0,
                                   target_count: int = 3166,
                                   augmentation_types: List[str] = ["eda", "llm"],
                                   random_state: int = 123,
                                   **data_kwargs) -> pd.DataFrame:
        """
        Run experiments comparing baseline, EDA, and LLM augmentation for underperforming labels.
        
        Args:
            underperforming_labels: List of labels to augment
            reduction_pct: Downsampling percentage
            target_count: Target number of examples per label after augmentation
            augmentation_types: List of augmentation types to test ["eda", "llm", "both"]
            random_state: Random seed
            **data_kwargs: Arguments for data preparation
            
        Returns:
            DataFrame with experiment results
        """
        print("="*80)
        print("AUGMENTATION EXPERIMENTS")
        print("="*80)
        print(f"Testing labels: {underperforming_labels}")
        print(f"Augmentation types: {augmentation_types}")
        print(f"Target count per label: {target_count}")
        print("="*80)
        
        # Prepare initial data
        self.prepare_data(**data_kwargs)
        train_df, val_df, test_df = self.data_loader.get_raw_dataframes(
            data_kwargs.get('train_ratio', 0.6),
            data_kwargs.get('val_ratio', 0.2), 
            data_kwargs.get('test_ratio', 0.2)
        )
        
        # Downsample training data once
        down_train = self.data_loader.downsample_by_label_reduction(
            train_df, reduction_pct=reduction_pct, random_state=random_state
        )
        
        metrics_list = []
        
        for label in underperforming_labels:
            print(f"\n{'='*60}")
            print(f"TESTING LABEL: {label}")
            print('='*60)
            
            # Baseline: just downsampled data
            baseline_metrics = self._run_single_experiment(
                down_train, val_df, test_df, 
                experiment_name=f"{label}_baseline",
                **data_kwargs
            )
            baseline_metrics.update({
                "target_label": label,
                "augmentation_type": "baseline",
                "reduction_pct": reduction_pct
            })
            metrics_list.append(baseline_metrics)
            
            # EDA augmentation
            if "eda" in augmentation_types:
                print(f"\n--- EDA Augmentation for {label} ---")
                eda_synthetic = self.data_loader.eda_augment_labels(
                    down_train, [label], target_count, random_state=random_state
                )
                
                if not eda_synthetic.empty:
                    train_eda = pd.concat([down_train, eda_synthetic], ignore_index=True)
                    eda_metrics = self._run_single_experiment(
                        train_eda, val_df, test_df,
                        experiment_name=f"{label}_eda",
                        **data_kwargs
                    )
                    eda_metrics.update({
                        "target_label": label,
                        "augmentation_type": "eda", 
                        "reduction_pct": reduction_pct,
                        "synthetic_samples": len(eda_synthetic)
                    })
                    metrics_list.append(eda_metrics)
            
            # LLM augmentation
            if "llm" in augmentation_types:
                print(f"\n--- LLM Augmentation for {label} ---")
                llm_synthetic = self.data_loader.llm_augment_labels(
                    down_train, [label], target_count, random_state=random_state
                )
                
                if not llm_synthetic.empty:
                    train_llm = pd.concat([down_train, llm_synthetic], ignore_index=True)
                    llm_metrics = self._run_single_experiment(
                        train_llm, val_df, test_df,
                        experiment_name=f"{label}_llm", 
                        **data_kwargs
                    )
                    llm_metrics.update({
                        "target_label": label,
                        "augmentation_type": "llm",
                        "reduction_pct": reduction_pct,
                        "synthetic_samples": len(llm_synthetic)
                    })
                    metrics_list.append(llm_metrics)
        
        # Create results DataFrame
        results_df = pd.DataFrame(metrics_list)
        
        # Print comparison summary
        self._print_augmentation_summary(results_df)
        
        return results_df
    
    def _run_single_experiment(self, train_df, val_df, test_df, experiment_name, **data_kwargs):
        """Run a single training experiment and return metrics."""
        
        # Convert to HF datasets and tokenize
        train_ds = self.data_loader.convert_to_hf_dataset(train_df)
        val_ds = self.data_loader.convert_to_hf_dataset(val_df)
        test_ds = self.data_loader.convert_to_hf_dataset(test_df)
        
        tokenizer_name = data_kwargs.get('tokenizer_name', self.model_name)
        max_length = data_kwargs.get('max_length', 128)
        
        train_ds = self.data_loader.tokenize_dataset(train_ds, tokenizer_name, max_length)
        val_ds = self.data_loader.tokenize_dataset(val_ds, tokenizer_name, max_length)
        test_ds = self.data_loader.tokenize_dataset(test_ds, tokenizer_name, max_length)
        
        # Create fresh model
        self.model_wrapper = MultiLabelBERT(self.model_name, self.num_labels)
        
        # Setup training
        training_args = self.setup_training_args(
            run_name=f"goemotions-{experiment_name}",
            output_dir=f"./results/{experiment_name}"
        )
        
        # Update datasets
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        
        # Train
        print(f"\nTraining {experiment_name} on {len(train_df)} samples...")
        self.create_trainer(training_args)
        self.trainer.train()
        
        # Evaluate
        print(f"Evaluating {experiment_name}...")
        val_metrics, thresholds = self.evaluate_and_tune_thresholds()
        test_metrics = self.evaluate_test_set()
        
        return {
            'train_samples': len(train_df),
            **{f"test_{k}": v for k, v in test_metrics.items()}
        }
    
    def _print_augmentation_summary(self, results_df):
        """Print summary of augmentation experiment results."""
        
        print(f"\n{'='*80}")
        print("AUGMENTATION EXPERIMENT SUMMARY")
        print('='*80)
        
        for label in results_df['target_label'].unique():
            label_results = results_df[results_df['target_label'] == label]
            
            print(f"\nLabel: {label}")
            print("-" * 40)
            
            for _, row in label_results.iterrows():
                aug_type = row['augmentation_type']
                micro_f1 = row['test_micro/f1']
                macro_f1 = row['test_macro/f1']
                samples = row.get('synthetic_samples', 0)
                
                if aug_type == 'baseline':
                    print(f"  Baseline:     Micro F1={micro_f1:.4f}, Macro F1={macro_f1:.4f}")
                else:
                    print(f"  {aug_type.upper():12s}: Micro F1={micro_f1:.4f}, Macro F1={macro_f1:.4f} (+{samples} synthetic)")
        
        print('='*80)