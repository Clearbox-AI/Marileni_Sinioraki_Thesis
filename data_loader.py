"""
Data loading utilities for GoEmotions multi-label emotion classification.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import os


class GoEmotionsDataLoader:
    """Data loader for preprocessed GoEmotions dataset."""
    
    def __init__(self, data_path: str = "data/processed_goemotions.csv"):
        self.data_path = data_path
        self.emotion_cols = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise'
        ]
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the preprocessed dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please run prepare_data.py first.")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with {len(self.df)} samples")
        return self.df
    
    def get_texts_and_labels(self) -> Tuple[List[str], np.ndarray]:
        """Extract texts and multi-label arrays."""
        if self.df is None:
            self.load_data()
        
        texts = self.df['text'].tolist()
        labels = self.df[self.emotion_cols].values.astype(np.float32)
        
        return texts, labels
    
    def get_label_statistics(self) -> dict:
        """Get statistics about label distribution."""
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_samples': len(self.df),
            'num_emotions': len(self.emotion_cols),
            'emotion_frequencies': self.df[self.emotion_cols].sum().to_dict(),
            'avg_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).mean(),
            'max_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).max(),
            'min_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).min()
        }
        
        return stats
    
    def get_emotion_names(self) -> List[str]:
        """Get list of emotion names."""
        return self.emotion_cols
    
    def get_sample_by_id(self, sample_id: int) -> dict:
        """Get a specific sample by ID."""
        if self.df is None:
            self.load_data()
        
        sample = self.df[self.df['id'] == sample_id]
        if len(sample) == 0:
            return None
        
        sample = sample.iloc[0]
        return {
            'id': sample['id'],
            'text': sample['text'],
            'subreddit': sample['subreddit'],
            'emotions': [emotion for emotion in self.emotion_cols if sample[emotion] == 1]
        }
    
    def filter_by_emotions(self, emotions: List[str]) -> pd.DataFrame:
        """Filter dataset to include only samples with specific emotions."""
        if self.df is None:
            self.load_data()
        
        # Check if all emotions are valid
        invalid_emotions = [e for e in emotions if e not in self.emotion_cols]
        if invalid_emotions:
            raise ValueError(f"Invalid emotions: {invalid_emotions}")
        
        # Filter samples that have at least one of the specified emotions
        mask = self.df[emotions].sum(axis=1) > 0
        filtered_df = self.df[mask]
        
        print(f"Filtered dataset: {len(filtered_df)} samples with emotions {emotions}")
        return filtered_df
    
    def get_subreddit_statistics(self) -> dict:
        """Get statistics about subreddit distribution."""
        if self.df is None:
            self.load_data()
        
        subreddit_counts = self.df['subreddit'].value_counts()
        
        return {
            'unique_subreddits': self.df['subreddit'].nunique(),
            'most_common_subreddit': subreddit_counts.index[0],
            'most_common_count': subreddit_counts.iloc[0],
            'least_common_count': subreddit_counts.min(),
            'avg_posts_per_subreddit': subreddit_counts.mean()
        }