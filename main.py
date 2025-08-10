"""
Main script for GoEmotions multi-label emotion classification.
Orchestrates the complete pipeline from data preparation to model training and evaluation.
"""

import argparse
import os
import sys
from prepare_data import GoEmotionsDataPreparator
from data_loader import GoEmotionsDataLoader
from trainer import GoEmotionsTrainingPipeline
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
    parser.add_argument('--train', action='store_true',
                       help='Train the BERT model')
    parser.add_argument('--wandb-key', type=str, 
                       help='Weights & Biases API key for experiment tracking')
    parser.add_argument('--data-dir', type=str, default='data/full_dataset/',
                       help='Directory to store raw dataset files')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                       help='Directory to store outputs and plots')
    parser.add_argument('--model-dir', type=str, default='./results/',
                       help='Directory to store model results')
    parser.add_argument('--model-name', type=str, default='bert-base-uncased',
                       help='HuggingFace model name')
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Create necessary directories if they don't exist
    create_directories([args.data_dir, args.output_dir, args.model_dir, 'data/', './logs/'])
    
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
    
    # Model training pipeline
    if args.train:
        print("\nStarting model training pipeline...")
        
        # Initialize training pipeline
        training_pipeline = GoEmotionsTrainingPipeline(
            model_name=args.model_name,
            output_dir=args.model_dir,
            num_labels=27
        )
        
        # Setup training arguments
        training_args = training_pipeline.setup_training_args(
            learning_rate=args.learning_rate,
            train_batch_size=args.batch_size,
            eval_batch_size=32,
            num_epochs=args.epochs,
            run_name=f"goemotions-{args.model_name.replace('/', '-')}"
        )
        
        # Run complete training pipeline
        results = training_pipeline.run_complete_pipeline(
            training_args=training_args,
            tokenizer_name=args.model_name,
            max_length=128,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        print(f"\nTraining complete. Model saved to {args.model_dir}")
        print(f"Best validation F1: {results['val_metrics']['micro/f1']:.4f}")
        print(f"Test F1: {results['test_metrics']['micro/f1']:.4f}")


def run_full_pipeline():
    """Run the complete pipeline without command line arguments."""
    print("Running complete GoEmotions processing and training pipeline...")

    # Create directories if they don't exist
    create_directories(['data/full_dataset/', 'data/', 'outputs/', './results/', './logs/'])
    
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
    
    print("\nData pipeline complete!")
    
    # Ask user if they want to proceed with training
    response = input("\nDo you want to proceed with model training? (y/n): ").strip().lower()
    
    if response == 'y' or response == 'yes':
        print("\nStarting model training...")
        
        # Initialize training pipeline
        training_pipeline = GoEmotionsTrainingPipeline(
            model_name="bert-base-uncased",
            output_dir="./results/",
            num_labels=27
        )
        
        # Run training with default settings
        results = training_pipeline.run_complete_pipeline(
            max_length=128,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        print("\nComplete pipeline finished!")
        print(f"Final test F1 score: {results['test_metrics']['micro/f1']:.4f}")
    else:
        print("Training skipped. Data preparation complete!")
    
    return df


if __name__ == "__main__":
    # If no command line arguments provided, run full pipeline
    if len(sys.argv) == 1:
        run_full_pipeline()
    else:
        main()