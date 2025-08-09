"""
Main script for GoEmotions multi-label emotion classification.
Orchestrates the complete pipeline from data preparation to analysis.
"""

import argparse
import os
from prepare_data import GoEmotionsDataPreparator
from data_loader import GoEmotionsDataLoader
from utils import (print_dataset_statistics, print_subreddit_statistics, 
                   plot_emotion_distribution, create_directories, setup_wandb)


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description='GoEmotions Multi-label Emotion Classification')
    parser.add_argument('--download', action='store_true', 
                       help='Download the GoEmotions dataset')
    parser.add_argument('--prepare', action='store_true', 
                       help='Prepare and preprocess the dataset')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze the dataset statistics')
    parser.add_argument('--wandb-key', type=str, 
                       help='Weights & Biases API key for experiment tracking')
    parser.add_argument('--data-dir', type=str, default='data/full_dataset/',
                       help='Directory to store raw dataset files')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                       help='Directory to store outputs and plots')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories([args.data_dir, args.output_dir, 'data/'])
    
    # Setup W&B if API key provided
    if args.wandb_key:
        setup_wandb(args.wandb_key)
    
    # Data preparation pipeline
    if args.download or args.prepare:
        print("Starting data preparation pipeline...")
        preparator = GoEmotionsDataPreparator(data_dir=args.data_dir)
        
        if args.download:
            preparator.download_dataset()
        
        if args.prepare:
            df = preparator.prepare_full_dataset(download=args.download, plot=True)
            
            # Save processed dataset
            output_path = "data/processed_goemotions.csv"
            df.to_csv(output_path, index=False)
            print(f"Processed dataset saved to {output_path}")
    
    # Dataset analysis
    if args.analyze:
        print("\nStarting dataset analysis...")
        
        # Load data
        data_loader = GoEmotionsDataLoader()
        df = data_loader.load_data()
        
        # Get statistics
        dataset_stats = data_loader.get_label_statistics()
        subreddit_stats = data_loader.get_subreddit_statistics()
        
        # Print statistics
        print_dataset_statistics(dataset_stats)
        print_subreddit_statistics(subreddit_stats)
        
        # Create visualizations
        emotion_counts = df[data_loader.get_emotion_names()].sum().sort_values(ascending=False)
        plot_emotion_distribution(
            emotion_counts, 
            title="GoEmotions Dataset - Emotion Distribution",
            save_path=os.path.join(args.output_dir, "emotion_distribution.png")
        )
        
        print(f"\nAnalysis complete. Outputs saved to {args.output_dir}")


def run_full_pipeline():
    """Run the complete pipeline without command line arguments."""
    print("Running complete GoEmotions processing pipeline...")
    
    # Create directories
    create_directories(['data/full_dataset/', 'data/', 'outputs/'])
    
    # Initialize data preparator
    preparator = GoEmotionsDataPreparator()
    
    # Run complete preparation
    df = preparator.prepare_full_dataset(download=True, plot=True)
    
    # Save processed dataset
    output_path = "data/processed_goemotions.csv"
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")
    
    # Load and analyze
    data_loader = GoEmotionsDataLoader()
    data_loader.df = df  # Use the already processed data
    
    # Get and print statistics
    dataset_stats = data_loader.get_label_statistics()
    subreddit_stats = data_loader.get_subreddit_statistics()
    
    print_dataset_statistics(dataset_stats)
    print_subreddit_statistics(subreddit_stats)
    
    print("\nPipeline complete!")
    return df


if __name__ == "__main__":
    # If no command line arguments provided, run full pipeline
    import sys
    if len(sys.argv) == 1:
        run_full_pipeline()
    else:
        main()