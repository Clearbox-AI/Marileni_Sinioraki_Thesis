"""
Example of how to use the integrated downsampling and label analysis functionality.
"""

from trainer import GoEmotionsTrainingPipeline
from data_loader import GoEmotionsDataLoader


def example_downsampling():
    """Example of running downsampling experiments."""
    
    print("Method 1: Using GoEmotionsTrainingPipeline")
    pipeline = GoEmotionsTrainingPipeline()
    
    # Run downsampling experiments
    results = pipeline.run_downsampling_experiments(
        reduction_percentages=[30, 60, 90],  # Test 3 levels
        random_state=42
    )
    
    print(f"\nResults:\n{results}")
    

def example_label_analysis():
    """Example of the label analysis experiment using regression."""
    
    print("Method 2: Label Analysis with Regression")
    pipeline = GoEmotionsTrainingPipeline()
    
    # Run the analysis experiment
    underperforming_labels = pipeline.analyze_underperforming_labels_after_downsampling(
        reduction_pct=50.0,      # Downsample by 50%
        random_state=42
    )
    
    print(f"\nUnderperforming labels identified: {underperforming_labels}")
    

def example_single_downsample():
    """Example of downsampling a single dataset."""
    
    print("Method 3: Single downsampling")
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


def example_augmentation_experiment():
    """Example of the complete augmentation experiment."""
    
    print("Method 4: Augmentation Experiment (EDA + LLM)")
    pipeline = GoEmotionsTrainingPipeline()
    
    # First identify underperforming labels
    underperforming_labels = pipeline.analyze_underperforming_labels_after_downsampling(
        reduction_pct=60.0,
        random_state=42
    )
    
    if underperforming_labels:
        print(f"\nRunning augmentation experiments for: {underperforming_labels}")
        
        # Run augmentation experiments  
        results = pipeline.run_augmentation_experiments(
            underperforming_labels=underperforming_labels[:2],  # Test first 2 labels
            reduction_pct=60.0,
            target_count=3166,
            augmentation_types=["eda", "llm"],
            random_state=42
        )
        
        print(f"\nAugmentation experiment results:")
        print(results[['target_label', 'augmentation_type', 'test_micro/f1', 'test_macro/f1']])


if __name__ == "__main__":
    print("Available examples:")
    print("1. Single downsampling")
    print("2. Full downsampling experiment")
    print("3. Label analysis with regression")
    print("4. Augmentation experiment (EDA + LLM)")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    if choice == "1":
        example_single_downsample()
    elif choice == "2":
        response = input("This will train models. Continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            example_downsampling()
    elif choice == "3":
        response = input("This will train a model. Continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            example_label_analysis()
    elif choice == "4":
        response = input("This will train multiple models and use LLM. Continue? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            example_augmentation_experiment()
    else:
        print("Invalid choice. Running single downsampling example...")
        example_single_downsample()
