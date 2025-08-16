"""
Example script demonstrating how to use the downsampling functionality.
This script shows how to run downsampling experiments on the GoEmotions dataset.
"""

from data_loader import GoEmotionsDataLoader
from downsampling import DownsamplingExperiment, DatasetDownsampler
import pandas as pd


def example_single_downsample():
    """Example of how to downsample a single dataset."""
    print("="*60)
    print("EXAMPLE: Single Dataset Downsampling")
    print("="*60)
    
    # Load data
    data_loader = GoEmotionsDataLoader()
    train_df, val_df, test_df = data_loader.get_raw_dataframes()
    
    # Initialize downsampler
    downsampler = DatasetDownsampler(data_loader.emotion_cols)
    
    # Downsample training set by 50%
    downsampled_train = downsampler.downsample_by_label_reduction(
        df=train_df,
        reduction_pct=50.0,
        random_state=42
    )
    
    print(f"\nOriginal training set size: {len(train_df)}")
    print(f"Downsampled training set size: {len(downsampled_train)}")
    print(f"Reduction: {(1 - len(downsampled_train)/len(train_df))*100:.1f}%")


def example_downsampling_experiment():
    """Example of how to run a full downsampling experiment."""
    print("\n" + "="*60)
    print("EXAMPLE: Full Downsampling Experiment")
    print("="*60)
    
    # Load data
    data_loader = GoEmotionsDataLoader()
    train_df, val_df, test_df = data_loader.get_raw_dataframes()
    
    # Initialize experiment
    experiment = DownsamplingExperiment(
        data_loader=data_loader,
        model_name="bert-base-uncased",
        output_dir="./results/downsampling_example"
    )
    
    # Run experiments with just a few reduction levels for demonstration
    results_df = experiment.run_downsampling_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        reduction_percentages=[30, 60, 90],  # Just 3 levels for demo
        random_state=123
    )
    
    # Show results
    print("\nExperiment Results:")
    print(results_df[['train_samples', 'test_micro/f1', 'test_macro/f1']].round(4))
    
    # Plot results
    experiment.plot_results(results_df, save_path="outputs/example_downsampling.png")
    
    return results_df


def example_custom_reduction_levels():
    """Example with custom reduction levels."""
    print("\n" + "="*60)
    print("EXAMPLE: Custom Reduction Levels")
    print("="*60)
    
    # Load data
    data_loader = GoEmotionsDataLoader()
    train_df, val_df, test_df = data_loader.get_raw_dataframes()
    
    # Initialize experiment
    experiment = DownsamplingExperiment(
        data_loader=data_loader,
        model_name="bert-base-uncased"
    )
    
    # Custom reduction levels focusing on small reductions
    custom_levels = [5, 15, 25, 35, 45]
    
    results_df = experiment.run_downsampling_experiment(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        reduction_percentages=custom_levels
    )
    
    print(f"\nCustom experiment with levels: {custom_levels}")
    print("Results:")
    print(results_df[['train_samples', 'test_micro/f1', 'test_macro/f1']].round(4))


if __name__ == "__main__":
    # Run all examples
    try:
        example_single_downsample()
        
        # Ask user if they want to run the training experiments
        response = input("\nDo you want to run the training experiments? (This will take a while) (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            example_downsampling_experiment()
            example_custom_reduction_levels()
        else:
            print("Training experiments skipped.")
            
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have prepared the dataset first by running:")
        print("python main.py --download --prepare")
