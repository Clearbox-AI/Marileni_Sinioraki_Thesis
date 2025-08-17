"""
BERT-based model for multi-label emotion classification with GoEmotions dataset.
"""

import torch.nn as nn
import numpy as np
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, classification_report, precision_recall_curve
import pandas as pd
from typing import List, Optional


class MultiLabelBERT:
    """
    Wrapper class for BERT multi-label classification model.
    Handles model initialization, threshold tuning, and evaluation metrics.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 27):
        """
        Initialize BERT model for multi-label classification.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of emotion labels (27 for GoEmotions without neutral)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.emotion_cols = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise'
        ]
        
        # Initialize model
        self.model = self._create_model()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.thresholds = None
        
    def _create_model(self):
        """Create and configure BERT model for multi-label classification."""
        model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels, # output 27 numbers (one per emotion) instead of default single classification
            problem_type="multi_label_classification" # Loss function: Uses BCEWithLogitsLoss instead of CrossEntropyLoss, Output activation: Uses sigmoid (can output multiple 1s) instead of softmax (only one 1)
        )
        
        print(f"Initialized {self.model_name} for multi-label classification")
        print(f"Classifier head: {model.classifier}")
        
        return model
    
    def tune_thresholds(self, logits: np.ndarray, labels: np.ndarray) -> List[float]:
        """
        Tune optimal thresholds for each emotion label using validation data.
        Uses F1-score maximization for each label independently.
        
        Args:
            logits: Model output logits
            labels: True labels
            
        Returns:
            List of optimal thresholds for each emotion
        """
        # Convert logits to probabilities
        probs = 1 / (1 + np.exp(-logits)) # Sigmoid: Converts any number to 0-1 range (probability)
        thresholds = []
        
        print("Tuning thresholds for optimal F1 scores...")
        
        for i in range(probs.shape[1]): # For each of the 27 emotions
            # Get precision-recall curve for this label
            precision, recall, threshold = precision_recall_curve(labels[:, i], probs[:, i]) #  Tests EVERY possible threshold from 0 to 1
            
            # Calculate F1 scores
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8) # + 1e-8: Prevents division by zero
            
            # Find threshold that maximizes F1. Picks the threshold that gives the highest F1 score for this specific emotion.
            best_idx = np.argmax(f1_scores)
            best_threshold = threshold[best_idx] if best_idx < len(threshold) else 0.5
            
            thresholds.append(best_threshold)
            
            print(f"{self.emotion_cols[i]}: threshold={best_threshold:.3f}, F1={f1_scores[best_idx]:.3f}")
        
        self.thresholds = thresholds
        return thresholds
    
    def compute_metrics(self, eval_pred, thresholds: Optional[List[float]] = None) -> dict:
        """
        Compute comprehensive evaluation metrics for multi-label classification.
        
        Args:
            eval_pred: Tuple of (logits, labels)
            thresholds: Optional list of thresholds (uses 0.5 if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        
        # Use provided thresholds or default to 0.5
        if thresholds is None:
            thresholds = [0.5] * self.num_labels
        
        # Apply thresholds to get predictions
        preds = np.zeros_like(probs)
        for i in range(self.num_labels):
            preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
        
        # Calculate micro and macro metrics
        micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        
        return {
            "micro/precision": micro_p,
            "micro/recall": micro_r,
            "micro/f1": micro_f1,
            "macro/precision": macro_p,
            "macro/recall": macro_r,
            "macro/f1": macro_f1
        }
    
    def print_class_report(self, logits: np.ndarray, labels: np.ndarray, 
                          thresholds: List[float], sort_by: str = "f1-score") -> pd.DataFrame:
        """
        Generate and print detailed per-class classification report.
        
        Args:
            logits: Model predictions (logits)
            labels: True labels
            thresholds: Threshold for each emotion
            sort_by: Metric to sort classes by
            
        Returns:
            DataFrame with per-class metrics
        """
        probs = 1 / (1 + np.exp(-logits))
        
        # Apply thresholds
        preds = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            preds[:, i] = (probs[:, i] > thresholds[i]).astype(int)
        
        # Generate classification report
        report = classification_report(
            labels, preds, 
            target_names=self.emotion_cols, 
            output_dict=True, 
            zero_division=0
        )
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(report).T
        df = df[df.index.isin(self.emotion_cols)]  # Filter out aggregated metrics
        df_sorted = df.sort_values(by=sort_by, ascending=False)
        
        print("Per-class Performance Report:")
        print("=" * 50)
        print(df_sorted.round(3))
        
        return df_sorted
    
    def identify_underperforming_labels(self, logits: np.ndarray, labels: np.ndarray, 
                                      thresholds: List[float]) -> List[str]:
        """
        Identify underperforming labels using regression analysis on F1 vs support.
        
        Args:
            logits: Model predictions (logits)
            labels: True labels
            thresholds: Threshold for each emotion
            
        Returns:
            List of emotion names that are underperforming (below regression line)
        """
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        
        # Get per-class report
        df_report = self.print_class_report(logits, labels, thresholds, sort_by="support")
        
        # Prepare data for regression analysis
        data = {
            'label': df_report.index.tolist(),
            'f1': df_report['f1-score'].values,
            'support': df_report['support'].values
        }
        
        df = pd.DataFrame(data)
        
        # Prepare features and targets for regression
        X = df["support"].values.reshape(-1, 1)
        y = df["f1"].values
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        # Add regression predictions to DataFrame
        df["f1_pred"] = y_pred
        df["above_line"] = df["f1"] >= df["f1_pred"]
        df["underperforming"] = ~df["above_line"]
        
        # Plot regression analysis
        plt.figure(figsize=(12, 8))
        
        # Scatter points based on whether they're above or below the regression line
        for _, row in df.iterrows():
            color = 'blue' if row["above_line"] else 'red'
            plt.scatter(row["support"], row["f1"], color=color, alpha=0.7, s=60)
            plt.annotate(row["label"], (row["support"], row["f1"]), 
                        fontsize=9, color=color, alpha=0.9, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Regression line
        sorted_idx = df["support"].argsort()
        plt.plot(df["support"].values[sorted_idx], y_pred[sorted_idx], 
                color='green', linestyle='-', linewidth=2, label='Linear Regression')
        
        # Plot settings
        plt.title('F1-score vs Support per Label (Red = Underperforming vs Regression)')
        plt.xlabel('Support (Number of examples)')
        plt.ylabel('F1-score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Get underperforming labels
        underperforming_labels = df[df["underperforming"]]["label"].tolist()
        
        # Print analysis summary
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS FOR LABEL PERFORMANCE")
        print("="*60)
        print(f"Regression R²: {reg.score(X, y):.3f}")
        print(f"Regression equation: F1 = {reg.coef_[0]:.6f} * support + {reg.intercept_:.3f}")
        
        print("\nUnderperforming labels (below regression line):")
        underperforming_df = df[df["underperforming"]].sort_values("f1", ascending=True)
        for _, row in underperforming_df.iterrows():
            diff = row["f1_pred"] - row["f1"]
            print(f"  {row['label']:12s}: F1={row['f1']:.3f}, Expected={row['f1_pred']:.3f}, Gap={diff:.3f}")
        
        print("\nWell-performing labels (above regression line):")
        performing_df = df[df["above_line"]].sort_values("f1", ascending=False)
        for _, row in performing_df.iterrows():
            diff = row["f1"] - row["f1_pred"]
            print(f"  {row['label']:12s}: F1={row['f1']:.3f}, Expected={row['f1_pred']:.3f}, Boost={diff:.3f}")
        
        print("="*60)
        
        return underperforming_labels


class MultilabelTrainer:
    """
    Custom trainer class that overrides loss computation for multi-label classification.
    Uses BCEWithLogitsLoss instead of CrossEntropyLoss.
    """
    
    def __init__(self, *args, **kwargs):
        # Import Trainer here to avoid dependency issues
        from transformers import Trainer
        
        # Initialize parent class
        self.trainer_class = Trainer
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def create_trainer(self, model, args, train_dataset, eval_dataset, tokenizer, compute_metrics_fn):
        """Create trainer with custom loss function."""
        
        class CustomTrainer(self.trainer_class):
            def __init__(self, loss_function, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_fn = loss_function
            
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss = self.loss_fn(logits, labels.float())
                return (loss, outputs) if return_outputs else loss
        
        return CustomTrainer(
            self.loss_fn,
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn
        )