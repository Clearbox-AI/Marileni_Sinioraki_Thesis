"""
Data loading and processing module for GoEmotions dataset.
"""

import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Define emotion label columns
EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

EMOTION_COLS_NO_NEUTRAL = [col for col in EMOTION_COLS if col != 'neutral']


class GoEmotionsDataLoader:
    """Data loader for the GoEmotions dataset."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Optional path to local CSV files. If None, loads from HuggingFace Hub.
        """
        self.data_path = data_path
        self.emotion_cols = EMOTION_COLS
        self.df = None
        
    def load_from_huggingface(self) -> Dataset:
        """Load GoEmotions dataset from HuggingFace Hub."""
        logger.info("Loading GoEmotions dataset from HuggingFace Hub...")
        dataset = load_dataset("go_emotions")
        logger.info(f"Dataset loaded: {dataset}")
        return dataset
    
    def load_from_csv(self, csv_paths: List[str]) -> pd.DataFrame:
        """
        Load and combine multiple CSV files.
        
        Args:
            csv_paths: List of paths to CSV files
            
        Returns:
            Combined DataFrame
        """
        logger.info(f"Loading data from CSV files: {csv_paths}")
        
        dfs = []
        for path in csv_paths:
            df = pd.read_csv(path)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {path}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_df)} rows")
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw dataset by removing unclear examples and processing labels.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning dataset...")
        initial_size = len(df)
        
        # Remove unclear examples
        if 'example_very_unclear' in df.columns:
            df = df[df["example_very_unclear"] == False]
            logger.info(f"Removed unclear examples: {initial_size} -> {len(df)}")
        
        # Group by ID and aggregate emotion labels
        grouped = df.groupby("id").agg({
            "text": "first",
            "subreddit": "first", 
            "created_utc": "first",
            **{col: "sum" for col in self.emotion_cols}
        }).reset_index()
        
        # Create multi-hot label vectors (1 if any rater selected that emotion)
        for col in self.emotion_cols:
            grouped[col] = (grouped[col] > 0).astype(int)
        
        # Remove posts with no emotion labels at all
        grouped["num_labels"] = grouped[self.emotion_cols].sum(axis=1)
        grouped = grouped[grouped["num_labels"] > 0]
        grouped.drop(columns=["num_labels"], inplace=True)
        
        logger.info(f"Final cleaned dataset size: {len(grouped)}")
        self.df = grouped
        return grouped
    
    def analyze_label_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze the distribution of emotion labels in the dataset.
        
        Args:
            df: DataFrame with emotion labels
            
        Returns:
            Dictionary mapping emotion names to counts
        """
        emotion_counts = df[self.emotion_cols].sum().sort_values(ascending=False)
        
        logger.info("Label distribution:")
        for emotion, count in emotion_counts.items():
            logger.info(f"{emotion}: {int(count)}")
        
        return emotion_counts.to_dict()
    
    def get_least_supported_labels(self, df: pd.DataFrame, n: int = 7) -> List[str]:
        """
        Get the n least supported emotion labels.
        
        Args:
            df: DataFrame with emotion labels
            n: Number of least supported labels to return
            
        Returns:
            List of emotion label names
        """
        label_support = df[self.emotion_cols].sum().sort_values()
        least_supported = label_support.head(n).index.tolist()
        
        logger.info(f"Least supported {n} labels: {least_supported}")
        return least_supported


class DataSplitter:
    """Handles different data splitting strategies."""
    
    @staticmethod
    def split_by_subreddit(df: pd.DataFrame, 
                          train_ratio: float = 0.6,
                          val_ratio: float = 0.2, 
                          test_ratio: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data maintaining subreddit distribution across splits.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data by subreddit...")
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for subreddit, group in df.groupby("subreddit"):
            # First split off test set
            train_val, test = train_test_split(
                group, 
                test_size=test_ratio,
                random_state=random_state,
                shuffle=True
            )
            
            # Then split train_val into train and validation
            val_size = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_size, 
                random_state=random_state,
                shuffle=True
            )
            
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
        
        # Combine all splits
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_df = pd.concat(val_dfs).reset_index(drop=True)
        test_df = pd.concat(test_dfs).reset_index(drop=True)
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    @staticmethod
    def split_with_unseen_subreddits(df: pd.DataFrame,
                                   test_subreddit_ratio: float = 0.2,
                                   val_ratio: float = 0.2,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with completely unseen subreddits in test set.
        
        Args:
            df: Input DataFrame
            test_subreddit_ratio: Proportion of subreddits for test set
            val_ratio: Proportion of remaining data for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data with unseen subreddits...")
        
        # Get unique subreddits and select test subreddits
        all_subreddits = df["subreddit"].unique()
        test_subreddits = pd.Series(all_subreddits).sample(
            frac=test_subreddit_ratio, 
            random_state=random_state
        )
        
        # Split based on subreddit presence
        test_df = df[df["subreddit"].isin(test_subreddits)].reset_index(drop=True)
        train_val_df = df[~df["subreddit"].isin(test_subreddits)].reset_index(drop=True)
        
        # Split remaining data into train/val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train subreddits: {train_df['subreddit'].nunique()}")
        logger.info(f"Val subreddits: {val_df['subreddit'].nunique()}")
        logger.info(f"Test subreddits: {test_df['subreddit'].nunique()}")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def split_by_timestamp(df: pd.DataFrame,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically by timestamp.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training (earliest posts)
            val_ratio: Proportion of remaining for validation
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data by timestamp...")
        
        # Sort by timestamp
        df_sorted = df.sort_values(by="created_utc").reset_index(drop=True)
        
        # Compute split indices
        train_split_idx = int(len(df_sorted) * train_ratio)
        
        # Split into train and test
        train_df = df_sorted.iloc[:train_split_idx].reset_index(drop=True)
        test_df = df_sorted.iloc[train_split_idx:].reset_index(drop=True)
        
        # Create validation set from training data
        train_df, val_df = train_test_split(
            train_df, 
            test_size=val_ratio,
            random_state=random_state
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df


def prepare_datasets_for_training(train_df: pd.DataFrame, 
                                val_df: pd.DataFrame,
                                test_df: pd.DataFrame,
                                emotion_cols: List[str] = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Convert pandas DataFrames to HuggingFace Dataset format.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        emotion_cols: List of emotion column names
        
    Returns:
        Tuple of HuggingFace Datasets (train, val, test)
    """
    if emotion_cols is None:
        emotion_cols = EMOTION_COLS
    
    logger.info("Converting DataFrames to HuggingFace Dataset format...")
    
    def convert_to_hf(df):
        # Drop existing 'labels' column if present
        df = df.drop(columns=["labels"], errors="ignore")
        
        hf_ds = Dataset.from_pandas(df)
        hf_ds = hf_ds.map(
            lambda x: {"labels": [x[col] for col in emotion_cols]}, 
            remove_columns=emotion_cols
        )
        return hf_ds
    
    train_ds = convert_to_hf(train_df)
    val_ds = convert_to_hf(val_df)
    test_ds = convert_to_hf(test_df)
    
    logger.info("Dataset conversion completed")
    return train_ds, val_ds, test_ds
