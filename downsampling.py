"""
Downsampling utilities for multi-label emotion classification datasets.
Provides functionality to reduce dataset size by label reduction and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, TrainingArguments
from model import MultiLabelBERT, MultilabelTrainer
from data_loader import GoEmotionsDataLoader


class DatasetDownsampler:
    """
    Handles downsampling of multi-label datasets using label reduction strategy.
    """
    
    def __init__(self, emotion_cols: List[str]):
        """
        Initialize downsampler with emotion column names.
        
        Args:
            emotion_cols: List of emotion column names
        """
        self.emotion_cols = emotion_cols
    
    def downsample_by_label_reduction(self,
                                    df: pd.DataFrame,
                                    reduction_pct: float,
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Downsample each emotion by a given percentage in a multi-label DataFrame.
        
        This method removes examples from the dataset while ensuring that each emotion
        is reduced by approximately the specified percentage. It uses a greedy approach
        to maintain label distribution while reducing overall dataset size.
        
        Args:
            df: Original DataFrame containing emotion columns
            reduction_pct: Percentage to remove from each label (0–100)
            random_state: Seed for reproducibility
            
        Returns:
            A new DataFrame with ~reduction_pct% of each emotion's examples removed
        """
        print(f"\nDownsampling dataset by {reduction_pct:.1f}% per label...")
        
        # 1. Compute original & target counts per label
        orig_counts = df[self.emotion_cols].sum().astype(int)
        target_counts = (orig_counts * (1 - reduction_pct / 100.0)).astype(int)
        
        # 2. Convert to numpy for efficient processing
        Y = df[self.emotion_cols].values.astype(int)
        current_counts = orig_counts.values.copy()
        target_vals = target_counts.values
        
        # 3. Track which samples to keep
        keep = np.ones(len(df), dtype=bool)
        rng = np.random.RandomState(random_state)
        
        # 4. Greedy removal strategy
        # Randomly iterate through samples and remove if it doesn't violate targets
        for idx in rng.permutation(len(df)):
            labels = np.where(Y[idx] == 1)[0]  # Get active emotion indices for this sample
            
            # Check if removing this sample keeps us above targets for all its emotions
            if np.all(current_counts[labels] - 1 >= target_vals[labels]):
                keep[idx] = False
                current_counts[labels] -= 1
                
                # Early stopping if all targets reached
                if np.all(current_counts <= target_vals):
                    break
        
        # 5. Create downsampled DataFrame
        down_df = df.iloc[keep].reset_index(drop=True)
        
        # 6. Print summary sorted by original frequency
        self._print_downsampling_summary(orig_counts, down_df, reduction_pct)
        
        return down_df
    
    def _print_downsampling_summary(self, 
                                   orig_counts: pd.Series,
                                   down_df: pd.DataFrame,
                                   reduction_pct: float):
        """Print detailed summary of downsampling results."""
        new_counts = down_df[self.emotion_cols].sum().astype(int)
        sorted_emotions = sorted(self.emotion_cols, 
                               key=lambda e: orig_counts.loc[e], 
                               reverse=True)
        
        print(f"\nDownsampling Summary (Target: {reduction_pct:.1f}% reduction):")
        print("=" * 60)
        print("Label         Original → New      (% Remaining)  (Actual Reduction)")
        print("-" * 60)
        
        total_orig = len(down_df) + (orig_counts.sum() - new_counts.sum())
        total_new = len(down_df)
        
        for emo in sorted_emotions:
            orig = orig_counts.loc[emo]
            new = new_counts.loc[emo]
            pct_remaining = (new / orig * 100) if orig > 0 else 0
            actual_reduction = (100 - pct_remaining) if orig > 0 else 0
            
            print(f"{emo:12s}: {orig:6d} → {new:6d}  ({pct_remaining:6.1f}%)     ({actual_reduction:5.1f}%)")
        
        print("-" * 60)
        dataset_reduction = ((total_orig - total_new) / total_orig * 100) if total_orig > 0 else 0
        print(f"Dataset size: {total_orig:6d} → {total_new:6d}  ({100-dataset_reduction:6.1f}%)     ({dataset_reduction:5.1f}%)")
        print("=" * 60)


class DownsamplingExperiment:
    """
    Conducts systematic downsampling experiments to analyze performance vs dataset size.
    """
    
    def __init__(self, 
                 data_loader: GoEmotionsDataLoader,
                 model_name: str = "bert-base-uncased",
                 output_dir: str = "./downsampling_results"):
        """
        Initialize downsampling experiment.
        
        Args:
            data_loader: Initialized GoEmotionsDataLoader
            model_name: HuggingFace model name
            output_dir: Directory to save results
        """
        self.data_loader = data_loader
        self.model_name = model_name
        self.output_dir = output_dir
        self.emotion_cols = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise'
        ]
        
        self.downsampler = DatasetDownsampler(self.emotion_cols)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def _prepare_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Convert DataFrames to HuggingFace datasets and tokenize."""
        
        def convert_to_hf_dataset(df):
            """Convert DataFrame to HuggingFace dataset format."""
            from datasets import Dataset
            
            # Create multi-hot encoded labels
            labels = df[self.emotion_cols].values.astype(float)
            
            dataset_dict = {
                'text': df['text'].tolist(),
                'labels': labels.tolist()
            }
            
            return Dataset.from_dict(dataset_dict)
        
        def tokenize_function(examples):
            """Tokenize text examples."""
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
        
        # Convert to HF datasets
        train_ds = convert_to_hf_dataset(train_df)
        val_ds = convert_to_hf_dataset(val_df)
        test_ds = convert_to_hf_dataset(test_df)
        
        # Tokenize
        train_ds = train_ds.map(tokenize_function, batched=True)
        val_ds = val_ds.map(tokenize_function, batched=True)
        test_ds = test_ds.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        
        return train_ds, val_ds, test_ds
    
    def _create_training_args(self) -> TrainingArguments:
        """Create optimized training arguments for downsampling experiments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=8,
            weight_decay=0.01,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_micro/f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="wandb",
            run_name="goemotions-downsampling-experiment"
        )
    
    def run_single_experiment(self,
                            train_df: pd.DataFrame,
                            val_df: pd.DataFrame,
                            test_df: pd.DataFrame,
                            reduction_pct: float,
                            random_state: int = 123) -> Dict:
        """
        Run a single downsampling experiment.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            reduction_pct: Percentage reduction to apply
            random_state: Random seed
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\n" + "="*80)
        print(f"RUNNING EXPERIMENT: {reduction_pct}% REDUCTION")
        print("="*80)
        
        # 1. Downsample training data
        down_train = self.downsampler.downsample_by_label_reduction(
            train_df, reduction_pct, random_state
        )
        
        # 2. Prepare datasets
        train_ds, val_ds, test_ds = self._prepare_datasets(down_train, val_df, test_df)
        
        # 3. Initialize fresh model
        model_wrapper = MultiLabelBERT(self.model_name, len(self.emotion_cols))
        trainer_wrapper = MultilabelTrainer()
        
        # 4. Setup training
        training_args = self._create_training_args()
        
        def compute_metrics_fn(eval_pred):
            return model_wrapper.compute_metrics(eval_pred)
        
        trainer = trainer_wrapper.create_trainer(
            model=model_wrapper.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            compute_metrics_fn=compute_metrics_fn
        )
        
        # 5. Train model
        print(f"\nTraining model on {len(down_train)} downsampled examples...")
        trainer.train()
        
        # 6. Tune thresholds on validation set
        print("\nTuning thresholds on validation set...")
        eval_output = trainer.predict(val_ds)
        val_logits = eval_output.predictions
        val_labels = eval_output.label_ids
        thresholds = model_wrapper.tune_thresholds(val_logits, val_labels)
        
        # 7. Evaluate on validation set
        val_metrics = model_wrapper.compute_metrics((val_logits, val_labels), thresholds)
        print("\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # 8. Evaluate on test set using tuned thresholds
        print("\nEvaluating on test set...")
        test_output = trainer.predict(test_ds)
        test_logits = test_output.predictions
        test_labels = test_output.label_ids
        
        test_metrics = model_wrapper.compute_metrics((test_logits, test_labels), thresholds)
        print("\nTest set metrics:")
        for key, value in test_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # 9. Generate detailed reports
        print("\n" + "-"*50)
        print("VALIDATION SET DETAILED REPORT:")
        print("-"*50)
        model_wrapper.print_class_report(val_logits, val_labels, thresholds)
        
        print("\n" + "-"*50) 
        print("TEST SET DETAILED REPORT:")
        print("-"*50)
        model_wrapper.print_class_report(test_logits, test_labels, thresholds)
        
        # 10. Prepare results
        result = {
            'reduction_pct': reduction_pct,
            'train_samples': len(down_train),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'thresholds': thresholds
        }
        
        # Add test metrics to result for easy access
        result.update({f"test_{k}": v for k, v in test_metrics.items()})
        
        return result
    
    def run_downsampling_experiment(self,
                                  train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  reduction_percentages: List[float] = None,
                                  random_state: int = 123) -> pd.DataFrame:
        """
        Run complete downsampling experiment across multiple reduction levels.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            reduction_percentages: List of reduction percentages to test
            random_state: Random seed
            
        Returns:
            DataFrame with aggregated results
        """
        if reduction_percentages is None:
            reduction_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        
        print(f"\n" + "="*100)
        print(f"STARTING DOWNSAMPLING EXPERIMENT")
        print(f"Testing reduction levels: {reduction_percentages}")
        print(f"Original training samples: {len(train_df)}")
        print("="*100)
        
        results_list = []
        
        for reduction_pct in reduction_percentages:
            try:
                result = self.run_single_experiment(
                    train_df, val_df, test_df, reduction_pct, random_state
                )
                results_list.append(result)
                
            except Exception as e:
                print(f"\nERROR in {reduction_pct}% experiment: {e}")
                continue
        
        # Create results DataFrame
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df = results_df.set_index('reduction_pct')
            
            print(f"\n" + "="*100)
            print("EXPERIMENT SUMMARY")
            print("="*100)
            print(results_df[['train_samples', 'test_micro/f1', 'test_macro/f1']].round(4))
            
            return results_df
        else:
            print("No successful experiments completed!")
            return pd.DataFrame()
    
    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """
        Plot downsampling experiment results.
        
        Args:
            results_df: Results DataFrame from run_downsampling_experiment
            save_path: Optional path to save the plot
        """
        if results_df.empty:
            print("No results to plot!")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Plot F1 scores
        plt.subplot(2, 2, 1)
        plt.plot(results_df.index, results_df["test_micro/f1"], 
                marker="o", label="Micro F1", linewidth=2, markersize=6)
        plt.plot(results_df.index, results_df["test_macro/f1"], 
                marker="s", label="Macro F1", linewidth=2, markersize=6)
        plt.xlabel("Downsampling Reduction (%)")
        plt.ylabel("F1 Score")
        plt.title("Test Set F1 vs. Training Set Downsampling")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot precision scores
        plt.subplot(2, 2, 2)
        plt.plot(results_df.index, results_df["test_micro/precision"], 
                marker="^", label="Micro Precision", linewidth=2, markersize=6)
        plt.plot(results_df.index, results_df["test_macro/precision"], 
                marker="v", label="Macro Precision", linewidth=2, markersize=6)
        plt.xlabel("Downsampling Reduction (%)")
        plt.ylabel("Precision")
        plt.title("Test Set Precision vs. Training Set Downsampling")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot recall scores
        plt.subplot(2, 2, 3)
        plt.plot(results_df.index, results_df["test_micro/recall"], 
                marker="d", label="Micro Recall", linewidth=2, markersize=6)
        plt.plot(results_df.index, results_df["test_macro/recall"], 
                marker="p", label="Macro Recall", linewidth=2, markersize=6)
        plt.xlabel("Downsampling Reduction (%)")
        plt.ylabel("Recall")
        plt.title("Test Set Recall vs. Training Set Downsampling")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot training samples
        plt.subplot(2, 2, 4)
        plt.plot(results_df.index, results_df["train_samples"], 
                marker="h", color="purple", linewidth=2, markersize=6)
        plt.xlabel("Downsampling Reduction (%)")
        plt.ylabel("Training Samples")
        plt.title("Training Set Size vs. Downsampling")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*80)
        print("DOWNSAMPLING EXPERIMENT SUMMARY")
        print("="*80)
        print(f"Best Micro F1: {results_df['test_micro/f1'].max():.4f} at {results_df['test_micro/f1'].idxmax()}% reduction")
        print(f"Best Macro F1: {results_df['test_macro/f1'].max():.4f} at {results_df['test_macro/f1'].idxmax()}% reduction")
        print(f"F1 degradation from 0% to 90%: {results_df.loc[10, 'test_micro/f1'] - results_df.loc[90, 'test_micro/f1']:.4f} (micro)")
        print("="*80)
