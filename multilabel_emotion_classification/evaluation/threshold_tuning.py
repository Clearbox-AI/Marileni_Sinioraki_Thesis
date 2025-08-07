"""Threshold tuning for multilabel classification."""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThresholdTuner:
    """Threshold tuner for multilabel classification."""
    
    def __init__(
        self,
        metric: str = 'f1_macro',
        search_method: str = 'grid',
        n_thresholds: int = 100,
        threshold_range: Tuple[float, float] = (0.1, 0.9)
    ):
        """
        Initialize threshold tuner.
        
        Args:
            metric: Metric to optimize ('f1_macro', 'f1_micro', 'precision_macro', etc.)
            search_method: Search method ('grid', 'random')
            n_thresholds: Number of thresholds to try
            threshold_range: Range of thresholds to search
        """
        self.metric = metric
        self.search_method = search_method
        self.n_thresholds = n_thresholds
        self.threshold_range = threshold_range
        self.best_thresholds = None
        self.best_score = None
        self.optimization_history = []
    
    def find_optimal_thresholds(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        per_class: bool = False
    ) -> Union[float, List[float]]:
        """
        Find optimal thresholds for multilabel classification.
        
        Args:
            y_true: True binary labels
            y_prob: Prediction probabilities
            per_class: Whether to find per-class thresholds
            
        Returns:
            Optimal threshold(s)
        """
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ImportError:
            raise ImportError("scikit-learn is required for threshold tuning")
        
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        if per_class:
            return self._find_per_class_thresholds(y_true, y_prob)
        else:
            return self._find_global_threshold(y_true, y_prob)
    
    def _find_global_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find a single optimal threshold for all classes."""
        thresholds = np.linspace(
            self.threshold_range[0], 
            self.threshold_range[1], 
            self.n_thresholds
        )
        
        best_threshold = 0.5
        best_score = -np.inf
        
        for threshold in thresholds:
            y_pred = (y_prob > threshold).astype(int)
            score = self._compute_metric(y_true, y_pred)
            
            self.optimization_history.append({
                'threshold': threshold,
                'score': score,
                'type': 'global'
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.best_thresholds = best_threshold
        self.best_score = best_score
        
        logger.info(f"Best global threshold: {best_threshold:.4f} with {self.metric}: {best_score:.4f}")
        
        return best_threshold
    
    def _find_per_class_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
        """Find optimal thresholds for each class separately."""
        n_classes = y_true.shape[1]
        best_thresholds = []
        
        thresholds = np.linspace(
            self.threshold_range[0], 
            self.threshold_range[1], 
            self.n_thresholds
        )
        
        for class_idx in range(n_classes):
            best_threshold = 0.5
            best_score = -np.inf
            
            class_true = y_true[:, class_idx]
            class_prob = y_prob[:, class_idx]
            
            for threshold in thresholds:
                class_pred = (class_prob > threshold).astype(int)
                
                # Create full prediction array for metric computation
                y_pred_full = np.zeros_like(y_true)
                y_pred_full[:, class_idx] = class_pred
                
                score = self._compute_class_metric(class_true, class_pred)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            best_thresholds.append(best_threshold)
            
            self.optimization_history.append({
                'class_idx': class_idx,
                'threshold': best_threshold,
                'score': best_score,
                'type': 'per_class'
            })
        
        self.best_thresholds = best_thresholds
        self.best_score = self._compute_metric(
            y_true, 
            self._apply_per_class_thresholds(y_prob, best_thresholds)
        )
        
        logger.info(f"Best per-class thresholds found with overall {self.metric}: {self.best_score:.4f}")
        
        return best_thresholds
    
    def _apply_per_class_thresholds(
        self, 
        y_prob: np.ndarray, 
        thresholds: List[float]
    ) -> np.ndarray:
        """Apply per-class thresholds to probabilities."""
        y_pred = np.zeros_like(y_prob)
        
        for class_idx, threshold in enumerate(thresholds):
            y_pred[:, class_idx] = (y_prob[:, class_idx] > threshold).astype(int)
        
        return y_pred
    
    def _compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the optimization metric."""
        try:
            from sklearn.metrics import (
                f1_score, precision_score, recall_score, 
                accuracy_score, hamming_loss
            )
        except ImportError:
            raise ImportError("scikit-learn is required for metrics computation")
        
        # Handle empty predictions
        if y_pred.sum() == 0:
            return 0.0
        
        if self.metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.metric == 'f1_micro':
            return f1_score(y_true, y_pred, average='micro', zero_division=0)
        elif self.metric == 'f1_weighted':
            return f1_score(y_true, y_pred, average='weighted', zero_division=0)
        elif self.metric == 'precision_macro':
            return precision_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.metric == 'precision_micro':
            return precision_score(y_true, y_pred, average='micro', zero_division=0)
        elif self.metric == 'recall_macro':
            return recall_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.metric == 'recall_micro':
            return recall_score(y_true, y_pred, average='micro', zero_division=0)
        elif self.metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.metric == 'hamming_loss':
            return -hamming_loss(y_true, y_pred)  # Negative because we want to maximize
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _compute_class_metric(self, y_true_class: np.ndarray, y_pred_class: np.ndarray) -> float:
        """Compute metric for a single class."""
        try:
            from sklearn.metrics import f1_score, precision_score, recall_score
        except ImportError:
            raise ImportError("scikit-learn is required for metrics computation")
        
        if y_pred_class.sum() == 0:
            return 0.0
        
        if 'f1' in self.metric:
            return f1_score(y_true_class, y_pred_class, zero_division=0)
        elif 'precision' in self.metric:
            return precision_score(y_true_class, y_pred_class, zero_division=0)
        elif 'recall' in self.metric:
            return recall_score(y_true_class, y_pred_class, zero_division=0)
        else:
            return f1_score(y_true_class, y_pred_class, zero_division=0)
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Args:
            save_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return
        
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        # Plot global threshold optimization
        global_history = [h for h in self.optimization_history if h.get('type') == 'global']
        
        if global_history:
            thresholds = [h['threshold'] for h in global_history]
            scores = [h['score'] for h in global_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, scores, 'b-', linewidth=2)
            plt.xlabel('Threshold')
            plt.ylabel(f'{self.metric.upper()}')
            plt.title('Threshold Optimization')
            plt.grid(True, alpha=0.3)
            
            # Mark best threshold
            if self.best_thresholds is not None and isinstance(self.best_thresholds, float):
                plt.axvline(x=self.best_thresholds, color='r', linestyle='--', 
                           label=f'Best: {self.best_thresholds:.3f}')
                plt.legend()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plot saved to: {save_path}")
            
            plt.show()


def find_optimal_thresholds(
    y_true: Union[np.ndarray, List[List[int]]],
    y_prob: Union[np.ndarray, List[List[float]]],
    metric: str = 'f1_macro',
    per_class: bool = False,
    n_thresholds: int = 100,
    threshold_range: Tuple[float, float] = (0.1, 0.9)
) -> Union[float, List[float]]:
    """
    Find optimal thresholds for multilabel classification.
    
    Args:
        y_true: True binary labels
        y_prob: Prediction probabilities
        metric: Metric to optimize
        per_class: Whether to find per-class thresholds
        n_thresholds: Number of thresholds to try
        threshold_range: Range of thresholds to search
        
    Returns:
        Optimal threshold(s)
    """
    tuner = ThresholdTuner(
        metric=metric,
        n_thresholds=n_thresholds,
        threshold_range=threshold_range
    )
    
    return tuner.find_optimal_thresholds(
        y_true=np.array(y_true),
        y_prob=np.array(y_prob),
        per_class=per_class
    )


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Union[float, List[float]],
    label_names: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate performance at given thresholds.
    
    Args:
        y_true: True binary labels
        y_prob: Prediction probabilities
        thresholds: Threshold(s) to evaluate
        label_names: Label names
        
    Returns:
        Dictionary containing evaluation results
    """
    from .metrics import compute_multilabel_metrics
    
    # Apply thresholds
    if isinstance(thresholds, float):
        y_pred = (y_prob > thresholds).astype(int)
    else:
        y_pred = np.zeros_like(y_prob)
        for i, threshold in enumerate(thresholds):
            y_pred[:, i] = (y_prob[:, i] > threshold).astype(int)
    
    # Compute metrics
    metrics = compute_multilabel_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        label_names=label_names
    )
    
    return {
        'thresholds': thresholds,
        'metrics': metrics,
        'predictions': y_pred
    }
