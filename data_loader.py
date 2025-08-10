"""
Data loading utilities for GoEmotions multi-label emotion classification.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import os
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer


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
    
    def split_by_subreddit(self, train_ratio: float = 0.6, val_ratio: float = 0.2, 
                          test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by subreddit to ensure no data leakage.
        Each subreddit's data is split proportionally across train/val/test.
        """
        if self.df is None:
            self.load_data()
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        print("Splitting dataset by subreddit...")
        
        for subreddit, group in self.df.groupby("subreddit"):
            # First split: separate test set
            train_val, test = train_test_split(
                group,
                test_size=test_ratio,
                random_state=42,
                shuffle=True
            )
            
            # Second split: divide remaining into train and validation
            val_size = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                random_state=42,
                shuffle=True
            )
            
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
        
        # Combine all splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_labels_for_hf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert emotion columns to HuggingFace format with 'labels' column."""
        df_copy = df.copy()
        df_copy["labels"] = df_copy[self.emotion_cols].values.tolist()
        return df_copy
    
    def convert_to_hf_dataset(self, df: pd.DataFrame) -> Dataset:
        """Convert pandas DataFrame to HuggingFace Dataset format."""
        # Prepare labels column
        df_with_labels = self.prepare_labels_for_hf(df)
        
        # Convert to HuggingFace Dataset
        hf_ds = Dataset.from_pandas(df_with_labels)
        
        # Remove individual emotion columns, keep 'labels' column
        hf_ds = hf_ds.map(
            lambda x: {"labels": [x[col] for col in self.emotion_cols]}, 
            remove_columns=self.emotion_cols
        )
        
        return hf_ds
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer_name: str = "bert-base-uncased", 
                        max_length: int = 128) -> Dataset:
        """
        Tokenize text data using specified tokenizer.
        
        Args:
            dataset: HuggingFace Dataset
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            "torch", 
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return tokenized_dataset
    
    def prepare_training_datasets(self, tokenizer_name: str = "bert-base-uncased",
                                max_length: int = 128, train_ratio: float = 0.6,
                                val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Complete pipeline to prepare tokenized datasets for training.
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds) as tokenized HuggingFace Datasets
        """
        # Split data by subreddit
        train_df, val_df, test_df = self.split_by_subreddit(train_ratio, val_ratio, test_ratio)
        
        # Print emotion statistics for training set
        print("\nTraining set emotion frequencies:")
        emotion_counts = train_df[self.emotion_cols].sum().sort_values(ascending=False)
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {int(count)}")
        
        # Convert to HuggingFace datasets
        train_ds = self.convert_to_hf_dataset(train_df)
        val_ds = self.convert_to_hf_dataset(val_df)
        test_ds = self.convert_to_hf_dataset(test_df)
        
        # Tokenize datasets
        train_ds = self.tokenize_dataset(train_ds, tokenizer_name, max_length)
        val_ds = self.tokenize_dataset(val_ds, tokenizer_name, max_length)
        test_ds = self.tokenize_dataset(test_ds, tokenizer_name, max_length)
        
        print(f"\nDatasets prepared with tokenizer: {tokenizer_name}")
        print(f"Max length: {max_length}")
        
        return train_ds, val_ds, test_ds