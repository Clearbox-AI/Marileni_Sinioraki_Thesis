"""
Data preparation script for GoEmotions multi-label emotion classification.
Downloads and preprocesses the GoEmotions dataset.
"""

import os
import subprocess # Running external commands
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple # For better code documentation


class GoEmotionsDataPreparator:
    """Handles downloading and preprocessing of GoEmotions dataset."""
    
    def __init__(self, data_dir: str = "data/full_dataset/"):
        self.data_dir = data_dir

        # The 27 emotions from GoEmotions (neutral removed)
        self.emotion_cols = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise'
        ]
        
    def download_dataset(self) -> None:
        """Download GoEmotions dataset files."""
        os.makedirs(self.data_dir, exist_ok=True) # Ensure data directory exists
        
        urls = [
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
        ]
        
        for url in urls:
            filename = url.split("/")[-1] # Gets the last part after the final slash
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                subprocess.run(["wget", "-P", self.data_dir, url], check=True) #  Raises error if download fails
            else:
                print(f"{filename} already exists, skipping download.")
    
    def load_and_combine_data(self) -> pd.DataFrame:
        """Load and combine the three CSV files."""
        print("Loading and combining dataset files...")
        
        df1 = pd.read_csv(os.path.join(self.data_dir, "goemotions_1.csv"))
        df2 = pd.read_csv(os.path.join(self.data_dir, "goemotions_2.csv"))
        df3 = pd.read_csv(os.path.join(self.data_dir, "goemotions_3.csv"))
        
        df = pd.concat([df1, df2, df3], ignore_index=True)
        print(f"Combined dataset shape: {df.shape}")
        return df
    
    def clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove unclear examples and neutral emotions."""
        print("Cleaning and filtering data...")
        
        # Remove unclear examples
        df = df[df["example_very_unclear"] == False]
        print(f"After removing unclear examples: {len(df)} samples")
        
        # Remove neutral examples
        df = df[df["neutral"] == False]
        print(f"After removing neutral examples: {len(df)} samples")
        
        return df
    
    def create_multilabel_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group by ID and create multi-hot label vectors."""
        print("Creating multi-label dataset...")
        
        # Each text was labeled by multiple people. 
        # Same ID appears multiple times with different emotion ratings.
        # Group by ID and aggregate
        grouped = df.groupby("id").agg({
            "text": "first", # Takes the text from the first row (they're all identical)
            "subreddit": "first",
            "created_utc": "first",
            **{col: "sum" for col in self.emotion_cols} # Sum the ratings for each emotion
        }).reset_index()
        
        # Create multi-hot label vector (1 only if all raters selected that emotion)
        for col in self.emotion_cols:
            # Get the mean - if all raters selected it, mean will be 1.0
            # If any rater didn't select it, mean will be < 1.0
            grouped[col] = (df.groupby("id")[col].mean() == 1.0).astype(int) # Converts True→1, False→0
        
        # Remove posts with no emotion label at all
        grouped["num_labels"] = grouped[self.emotion_cols].sum(axis=1)
        grouped = grouped[grouped["num_labels"] > 0]
        grouped.drop(columns=["num_labels"], inplace=True)
        
        # Select final columns
        final_df = grouped[["id", "text", "subreddit", "created_utc"] + self.emotion_cols]
        print(f"Final dataset length: {len(final_df)}")
        
        return final_df
    
    def analyze_dataset(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Analyze emotion distribution and subreddit statistics."""
        print("\n=== Dataset Analysis ===")
        
        # Emotion frequency analysis
        emotion_counts = df[self.emotion_cols].sum().sort_values(ascending=False)
        print("\nEmotion frequencies:")
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {int(count)}")
        
        # Subreddit analysis
        unique_subreddits = df['subreddit'].nunique()
        print(f"\nUnique subreddits: {unique_subreddits}")
        
        subreddit_counts = df['subreddit'].value_counts()
        print("\nTop 10 most frequent subreddits:")
        print(subreddit_counts.head(10))
        
        # Least common subreddits
        min_posts = subreddit_counts.min()
        num_with_min_posts = (subreddit_counts == min_posts).sum()
        print(f"\nMinimum posts in any subreddit: {min_posts}")
        print(f"Number of subreddits with {min_posts} posts: {num_with_min_posts}")
        
        return emotion_counts, subreddit_counts
    
    def plot_emotion_distribution(self, emotion_counts: pd.Series, save_path: str = None) -> None:
        """Plot emotion frequency distribution."""
        plt.figure(figsize=(12, 6))
        emotion_counts.plot(kind='bar', title="Emotion Frequency in GoEmotions Dataset")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def prepare_full_dataset(self, download: bool = True, plot: bool = True) -> pd.DataFrame:
        """Complete data preparation pipeline."""
        if download:
            self.download_dataset()
        
        # Load and process data
        df = self.load_and_combine_data()
        df = self.clean_and_filter_data(df)
        df = self.create_multilabel_dataset(df)
        
        # Analyze dataset
        emotion_counts, subreddit_counts = self.analyze_dataset(df)
        
        # Plot emotion distribution
        if plot:
            self.plot_emotion_distribution(emotion_counts)
        
        return df


def main():
    """Main function to run data preparation."""
    preparator = GoEmotionsDataPreparator()
    df = preparator.prepare_full_dataset()
    
    # Save processed dataset
    output_path = "data/processed_goemotions.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to {output_path}")
    
    return df


if __name__ == "__main__":
    main()