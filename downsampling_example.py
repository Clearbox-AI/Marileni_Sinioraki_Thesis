"""
Example of how to use the integrated downsampling functionality.
"""

from trainer import GoEmotionsTrainingPipeline
from data_loader import GoEmotionsDataLoader


def example_downsampling():
    """Example of running downsampling experiments."""
    
    # Method 1: Use the training pipeline (recommended)
    print("Method 1: Using GoEmotionsTrainingPipeline")
    pipeline = GoEmotionsTrainingPipeline()
    
    # Run downsampling experiments
    results = pipeline.run_downsampling_experiments(
        reduction_percentages=[30, 60, 90],  # Test 3 levels
        random_state=42
    )
    
    print(f"\nResults:\n{results}")
    
    
def example_single_downsample():
    """Example of downsampling a single dataset."""
    
    print("Method 2: Single downsampling")
    data_loader = GoEmotionsDataLoader()
    train_df, val_df, test_df = data_loader.get_raw_dataframes()
    
    # Downsample training set by 50%
    downsampled = data_loader.downsample_by_label_reduction(
        df=train_df,
        reduction_pct=50.0,
        random_state=42
    )
    
    print(f"Original: {len(train_df)} samples")
    print(f"Downsampled: {len(downsampled)} samples")


if __name__ == "__main__":
    # Run single downsampling example
    example_single_downsample()
    
    # Ask if user wants to run full experiment
    response = input("\nRun full downsampling experiment? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        example_downsampling()
