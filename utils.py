"""
Utility functions for GoEmotions multi-label emotion classification.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os


def setup_wandb(api_key: str = None, project_name: str = "goemotions-classification"):
    """Setup Weights & Biases for experiment tracking."""
    try:
        import wandb
        
        if api_key:
            wandb.login(key=api_key)
        
        wandb.init(project=project_name)
        print("W&B initialized successfully")
        return wandb
    except ImportError:
        print("wandb not installed. Install with: pip install wandb")
        return None


def plot_emotion_distribution(emotion_counts: pd.Series, title: str = "Emotion Distribution", 
                            figsize: tuple = (12, 6), save_path: str = None):
    """Plot emotion frequency distribution with improved styling."""
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bars = plt.bar(range(len(emotion_counts)), emotion_counts.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, len(emotion_counts))))
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Emotions", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # Set x-axis labels
    plt.xticks(range(len(emotion_counts)), emotion_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_label_distribution(labels: np.ndarray, emotion_names: List[str], 
                          title: str = "Label Co-occurrence Matrix"):
    """Plot label co-occurrence heatmap."""
    # Calculate co-occurrence matrix
    cooccurrence = np.dot(labels.T, labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cooccurrence, 
                xticklabels=emotion_names, 
                yticklabels=emotion_names,
                annot=True, 
                fmt='d',
                cmap='Blues',
                square=True)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def print_dataset_statistics(stats: Dict[str, Any]):
    """Print formatted dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Total samples: {stats['total_samples']:,}")
    print(f"Number of emotions: {stats['num_emotions']}")
    print(f"Average labels per sample: {stats['avg_labels_per_sample']:.2f}")
    print(f"Max labels per sample: {stats['max_labels_per_sample']}")
    print(f"Min labels per sample: {stats['min_labels_per_sample']}")
    
    print(f"\nTop 10 most frequent emotions:")
    emotion_freq = stats['emotion_frequencies']
    sorted_emotions = sorted(emotion_freq.items(), key=lambda x: x[1], reverse=True)
    
    for emotion, count in sorted_emotions[:10]:
        percentage = (count / stats['total_samples']) * 100
        print(f"  {emotion}: {count:,} ({percentage:.1f}%)")
    
    print("="*50)


def print_subreddit_statistics(stats: Dict[str, Any]):
    """Print formatted subreddit statistics."""
    print("\n" + "="*50)
    print("SUBREDDIT STATISTICS")
    print("="*50)
    
    print(f"Unique subreddits: {stats['unique_subreddits']:,}")
    print(f"Most common subreddit: {stats['most_common_subreddit']} ({stats['most_common_count']:,} posts)")
    print(f"Least common subreddit count: {stats['least_common_count']}")
    print(f"Average posts per subreddit: {stats['avg_posts_per_subreddit']:.1f}")
    
    print("="*50)


def create_directories(paths: List[str]):
    """Create directories if they don't exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"Directory created/verified: {path}")


def save_emotion_examples(df: pd.DataFrame, emotion: str, num_examples: int = 5, 
                         output_file: str = None):
    """Save example texts for a specific emotion."""
    if emotion not in df.columns:
        print(f"Emotion '{emotion}' not found in dataset")
        return
    
    emotion_samples = df[df[emotion] == 1]['text'].head(num_examples)
    
    output = f"\n=== Examples for emotion: {emotion.upper()} ===\n"
    for i, text in enumerate(emotion_samples, 1):
        output += f"{i}. {text}\n\n"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Examples saved to {output_file}")
    else:
        print(output)


def validate_emotion_labels(emotion_names: List[str]) -> bool:
    """Validate that emotion names are correct."""
    expected_emotions = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise'
    ]
    
    if set(emotion_names) != set(expected_emotions):
        missing = set(expected_emotions) - set(emotion_names)
        extra = set(emotion_names) - set(expected_emotions)
        
        if missing:
            print(f"Missing emotions: {missing}")
        if extra:
            print(f"Extra emotions: {extra}")
        return False
    
    return True