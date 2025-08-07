"""Metrics computation for multilabel emotion classification."""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class MultiLabelMetrics:
    """Container for multilabel classification metrics."""
    
    # Overall metrics
    f1_micro: float
    f1_macro: float
    f1_weighted: float
    f1_samples: float
    
    precision_micro: float
    precision_macro: float
    precision_weighted: float
    precision_samples: float
    
    recall_micro: float
    recall_macro: float
    recall_weighted: float
    recall_samples: float
    
    accuracy: float
    hamming_loss: float
    jaccard_score: float
    
    # Per-class metrics
    per_class_f1: List[float]
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_support: List[int]
    
    # Additional metrics
    exact_match_ratio: float
    
    def to_dict(self) -> Dict[str, Union[float, List[float], List[int]]]:
        """Convert metrics to dictionary."""
        return {
            'f1_micro': self.f1_micro,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'f1_samples': self.f1_samples,
            'precision_micro': self.precision_micro,
            'precision_macro': self.precision_macro,
            'precision_weighted': self.precision_weighted,
            'precision_samples': self.precision_samples,
            'recall_micro': self.recall_micro,
            'recall_macro': self.recall_macro,
            'recall_weighted': self.recall_weighted,
            'recall_samples': self.recall_samples,
            'accuracy': self.accuracy,
            'hamming_loss': self.hamming_loss,
            'jaccard_score': self.jaccard_score,
            'per_class_f1': self.per_class_f1,
            'per_class_precision': self.per_class_precision,
            'per_class_recall': self.per_class_recall,
            'per_class_support': self.per_class_support,
            'exact_match_ratio': self.exact_match_ratio,
        }


def compute_multilabel_metrics(
    y_true: Union[np.ndarray, List[List[int]]],
    y_pred: Union[np.ndarray, List[List[int]]],
    y_prob: Optional[Union[np.ndarray, List[List[float]]]] = None,
    label_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> MultiLabelMetrics:
    """
    Compute comprehensive multilabel classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels or probabilities
        y_prob: Prediction probabilities (optional)
        label_names: Names of the labels
        threshold: Threshold for converting probabilities to binary predictions
        
    Returns:
        MultiLabelMetrics object containing all computed metrics
    """
    try:
        from sklearn.metrics import (
            f1_score, precision_score, recall_score,
            accuracy_score, hamming_loss, jaccard_score,
            classification_report
        )
    except ImportError:
        raise ImportError("scikit-learn is required for metrics computation")
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    
    # Handle probabilities vs binary predictions
    if y_prob is not None:
        y_prob = np.array(y_prob)
        y_pred = (y_prob > threshold).astype(int)
    else:
        y_pred = np.array(y_pred)
        if y_pred.dtype == float:
            y_pred = (y_pred > threshold).astype(int)
    
    # Ensure binary format
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Handle empty predictions
    if y_pred.sum() == 0:
        warnings.warn("All predictions are zero. Adjusting metrics computation.")
        # Set at least one prediction to 1 to avoid division by zero
        y_pred_adjusted = y_pred.copy()
        y_pred_adjusted[0, 0] = 1
        y_pred = y_pred_adjusted
    
    # Compute F1 scores
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_samples = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Compute precision scores
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_samples = precision_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Compute recall scores
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_samples = recall_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Compute other metrics
    accuracy = accuracy_score(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_support = y_true.sum(axis=0).tolist()
    
    # Exact match ratio (all labels must be predicted correctly)
    exact_matches = np.all(y_true == y_pred, axis=1)
    exact_match_ratio = np.mean(exact_matches)
    
    return MultiLabelMetrics(
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        f1_samples=f1_samples,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        precision_samples=precision_samples,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        recall_samples=recall_samples,
        accuracy=accuracy,
        hamming_loss=hamming,
        jaccard_score=jaccard,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
        per_class_support=per_class_support,
        exact_match_ratio=exact_match_ratio
    )


def compute_metrics_from_predictions(predictions, labels, label_names=None):
    """
    Compute metrics from model predictions (for use with Trainer).
    
    Args:
        predictions: Model predictions
        labels: True labels
        label_names: Label names
        
    Returns:
        Dictionary of metrics
    """
    # Handle different prediction formats
    if hasattr(predictions, 'predictions'):
        y_pred = predictions.predictions
    else:
        y_pred = predictions
        
    # Convert logits to probabilities and then to binary predictions
    if y_pred.shape[-1] > 1:  # Multi-output
        y_prob = 1 / (1 + np.exp(-y_pred))  # Sigmoid
        y_pred_binary = (y_prob > 0.5).astype(int)
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
        y_prob = y_pred
    
    # Compute metrics
    metrics = compute_multilabel_metrics(
        y_true=labels,
        y_pred=y_pred_binary,
        y_prob=y_prob,
        label_names=label_names
    )
    
    # Return dictionary with key metrics for training
    return {
        'f1_macro': metrics.f1_macro,
        'f1_micro': metrics.f1_micro,
        'f1_weighted': metrics.f1_weighted,
        'precision_macro': metrics.precision_macro,
        'recall_macro': metrics.recall_macro,
        'accuracy': metrics.accuracy,
        'hamming_loss': metrics.hamming_loss,
        'exact_match_ratio': metrics.exact_match_ratio,
    }


def print_metrics_summary(metrics: MultiLabelMetrics, label_names: Optional[List[str]] = None):
    """
    Print a summary of the metrics.
    
    Args:
        metrics: Computed metrics
        label_names: Names of the labels
    """
    print("\n" + "="*60)
    print("MULTILABEL CLASSIFICATION METRICS SUMMARY")
    print("="*60)
    
    print(f"Overall Performance:")
    print(f"  F1 Score (Macro):     {metrics.f1_macro:.4f}")
    print(f"  F1 Score (Micro):     {metrics.f1_micro:.4f}")
    print(f"  F1 Score (Weighted):  {metrics.f1_weighted:.4f}")
    print(f"  F1 Score (Samples):   {metrics.f1_samples:.4f}")
    
    print(f"\n  Precision (Macro):    {metrics.precision_macro:.4f}")
    print(f"  Precision (Micro):    {metrics.precision_micro:.4f}")
    print(f"  Precision (Weighted): {metrics.precision_weighted:.4f}")
    
    print(f"\n  Recall (Macro):       {metrics.recall_macro:.4f}")
    print(f"  Recall (Micro):       {metrics.recall_micro:.4f}")
    print(f"  Recall (Weighted):    {metrics.recall_weighted:.4f}")
    
    print(f"\n  Accuracy:             {metrics.accuracy:.4f}")
    print(f"  Hamming Loss:         {metrics.hamming_loss:.4f}")
    print(f"  Jaccard Score:        {metrics.jaccard_score:.4f}")
    print(f"  Exact Match Ratio:    {metrics.exact_match_ratio:.4f}")
    
    # Per-class metrics
    if label_names is not None and len(label_names) == len(metrics.per_class_f1):
        print(f"\nPer-Class Performance:")
        print(f"{'Label':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
        print("-" * 60)
        
        for i, label in enumerate(label_names):
            print(f"{label:<20} {metrics.per_class_f1[i]:<8.4f} "
                  f"{metrics.per_class_precision[i]:<10.4f} "
                  f"{metrics.per_class_recall[i]:<8.4f} "
                  f"{metrics.per_class_support[i]:<8}")
    
    print("="*60)
