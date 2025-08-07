"""Evaluation reporting and result saving utilities."""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import logging

from .metrics import MultiLabelMetrics

logger = logging.getLogger(__name__)


def generate_classification_report(
    metrics: MultiLabelMetrics,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        metrics: Computed metrics
        label_names: Names of the labels
        save_path: Path to save the report
        
    Returns:
        Classification report as string
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(len(metrics.per_class_f1))]
    
    # Create report
    report_lines = []
    report_lines.append("MULTILABEL EMOTION CLASSIFICATION REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("OVERALL PERFORMANCE METRICS")
    report_lines.append("-" * 60)
    report_lines.append(f"F1 Score (Macro):      {metrics.f1_macro:.4f}")
    report_lines.append(f"F1 Score (Micro):      {metrics.f1_micro:.4f}")
    report_lines.append(f"F1 Score (Weighted):   {metrics.f1_weighted:.4f}")
    report_lines.append(f"F1 Score (Samples):    {metrics.f1_samples:.4f}")
    report_lines.append("")
    report_lines.append(f"Precision (Macro):     {metrics.precision_macro:.4f}")
    report_lines.append(f"Precision (Micro):     {metrics.precision_micro:.4f}")
    report_lines.append(f"Precision (Weighted):  {metrics.precision_weighted:.4f}")
    report_lines.append(f"Precision (Samples):   {metrics.precision_samples:.4f}")
    report_lines.append("")
    report_lines.append(f"Recall (Macro):        {metrics.recall_macro:.4f}")
    report_lines.append(f"Recall (Micro):        {metrics.recall_micro:.4f}")
    report_lines.append(f"Recall (Weighted):     {metrics.recall_weighted:.4f}")
    report_lines.append(f"Recall (Samples):      {metrics.recall_samples:.4f}")
    report_lines.append("")
    report_lines.append(f"Accuracy:              {metrics.accuracy:.4f}")
    report_lines.append(f"Hamming Loss:          {metrics.hamming_loss:.4f}")
    report_lines.append(f"Jaccard Score:         {metrics.jaccard_score:.4f}")
    report_lines.append(f"Exact Match Ratio:     {metrics.exact_match_ratio:.4f}")
    report_lines.append("")
    
    # Per-class metrics
    report_lines.append("PER-CLASS PERFORMANCE METRICS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Label':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
    report_lines.append("-" * 80)
    
    for i, label in enumerate(label_names):
        report_lines.append(
            f"{label:<25} {metrics.per_class_f1[i]:<8.4f} "
            f"{metrics.per_class_precision[i]:<10.4f} "
            f"{metrics.per_class_recall[i]:<8.4f} "
            f"{metrics.per_class_support[i]:<8}"
        )
    
    report_lines.append("-" * 80)
    
    # Summary statistics
    report_lines.append("")
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    report_lines.append(f"Total samples:         {sum(metrics.per_class_support)}")
    report_lines.append(f"Average support:       {np.mean(metrics.per_class_support):.1f}")
    report_lines.append(f"Min/Max F1:           {min(metrics.per_class_f1):.4f} / {max(metrics.per_class_f1):.4f}")
    report_lines.append(f"F1 Std Dev:           {np.std(metrics.per_class_f1):.4f}")
    
    report_text = "\n".join(report_lines)
    
    # Save report if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Classification report saved to: {save_path}")
    
    return report_text


def save_evaluation_results(
    metrics: MultiLabelMetrics,
    output_dir: str,
    label_names: Optional[List[str]] = None,
    model_name: str = "emotion_classifier",
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Save comprehensive evaluation results.
    
    Args:
        metrics: Computed metrics
        output_dir: Directory to save results
        label_names: Names of the labels
        model_name: Name of the model
        additional_info: Additional information to include
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics as JSON
    metrics_dict = metrics.to_dict()
    if additional_info:
        metrics_dict.update(additional_info)
    
    metrics_dict['timestamp'] = timestamp
    metrics_dict['model_name'] = model_name
    
    if label_names:
        metrics_dict['label_names'] = label_names
    
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, f"report_{timestamp}.txt")
    generate_classification_report(metrics, label_names, report_path)
    
    # Save per-class metrics as CSV
    if label_names:
        per_class_df = pd.DataFrame({
            'label': label_names,
            'f1_score': metrics.per_class_f1,
            'precision': metrics.per_class_precision,
            'recall': metrics.per_class_recall,
            'support': metrics.per_class_support
        })
        
        csv_path = os.path.join(output_dir, f"per_class_metrics_{timestamp}.csv")
        per_class_df.to_csv(csv_path, index=False)
        logger.info(f"Per-class metrics saved to: {csv_path}")
    
    # Create summary
    summary = {
        'model_name': model_name,
        'timestamp': timestamp,
        'f1_macro': metrics.f1_macro,
        'f1_micro': metrics.f1_micro,
        'accuracy': metrics.accuracy,
        'exact_match_ratio': metrics.exact_match_ratio,
        'files': {
            'metrics': metrics_path,
            'report': report_path,
            'per_class_csv': csv_path if label_names else None
        }
    }
    
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Evaluation summary saved to: {summary_path}")


def create_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
):
    """
    Create confusion matrix heatmap for multilabel classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        label_names: Names of the labels
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import multilabel_confusion_matrix
    except ImportError:
        logger.warning("matplotlib, seaborn, or sklearn not available, skipping heatmap")
        return
    
    # Compute multilabel confusion matrices
    cm_array = multilabel_confusion_matrix(y_true, y_pred)
    
    # Create subplots for each class
    n_classes = len(label_names)
    cols = min(4, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_classes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (cm, label) in enumerate(zip(cm_array, label_names)):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{label}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_classes, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix heatmap saved to: {save_path}")
    
    plt.show()


def create_performance_comparison_plot(
    results_list: List[Dict[str, Any]],
    metric_names: List[str] = ['f1_macro', 'f1_micro', 'precision_macro', 'recall_macro'],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8)
):
    """
    Create a performance comparison plot for multiple models/experiments.
    
    Args:
        results_list: List of results dictionaries
        metric_names: Names of metrics to compare
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping comparison plot")
        return
    
    if not results_list:
        logger.warning("No results to plot")
        return
    
    # Extract data
    model_names = [result.get('model_name', f'Model {i}') for i, result in enumerate(results_list)]
    
    # Prepare data for plotting
    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, metric in enumerate(metric_names):
        values = []
        for result in results_list:
            if isinstance(result.get('metrics'), MultiLabelMetrics):
                metrics_obj = result['metrics']
                value = getattr(metrics_obj, metric, 0)
            else:
                value = result.get('metrics', {}).get(metric, 0)
            values.append(value)
        
        bars = ax.bar(x + i * width - width * (len(metric_names) - 1) / 2, 
                     values, width, label=metric.replace('_', ' ').title())
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance comparison plot saved to: {save_path}")
    
    plt.show()


def load_evaluation_results(results_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Dictionary containing results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Evaluation results loaded from: {results_path}")
    return results
