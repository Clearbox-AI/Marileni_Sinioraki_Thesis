# GoEmotions Multi-Label Emotion Classification

A streamlined implementation of multi-label emotion classification using BERT on the GoEmotions dataset with experimentation capabilities including downsampling, regression analysis, and data augmentation.

## Features

- **Multi-label BERT Classification**: Fine-tuned BERT model for 27 emotion categories
- **Data Splitting**: Subreddit-stratified splits to prevent data leakage  
- **Downsampling Experiments**: Test model performance under data scarcity
- **Regression Analysis**: Identify underperforming emotion labels using statistical analysis
- **Data Augmentation**: EDA (synonym replacement) and LLM-based augmentation techniques
- **Automated Pipeline**: Complete CLI interface for all experiments

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
# Download, process data and train model
python main.py

# Or use specific commands
python main.py --prepare --analyze --train
```

### 3. Run Experiments
```bash
# Downsampling experiments
python main.py --downsample --reduction-levels 30 60 90

# Label analysis with regression
python main.py --analyze-labels

# Augmentation experiments
python main.py --augment-experiment --augmentation-types eda llm
```

## Core Files

- **`main.py`**: Command-line interface and pipeline orchestration
- **`data_loader.py`**: Data loading, preprocessing, and augmentation methods
- **`model.py`**: BERT model wrapper with threshold tuning and regression analysis
- **`trainer.py`**: Training pipeline with experiment management
- **`prepare_data.py`**: GoEmotions dataset download and preprocessing
- **`utils.py`**: Utility functions for visualization and statistics
- **`examples.py`**: Usage examples for different functionalities

## Experiments

### Downsampling Analysis
Tests model robustness under reduced training data:
```bash
python main.py --downsample --reduction-levels 10 30 50 70 90
```

### Underperforming Label Identification
Uses linear regression to identify emotions that perform poorly relative to their support:
```bash
python main.py --analyze-labels
```

### Data Augmentation
Compares baseline vs EDA vs LLM augmentation for underperforming labels:
```bash
python main.py --augment-experiment
```

## Dataset

GoEmotions: 58k Reddit comments labeled with 27 emotions
- Automatic download and preprocessing
- Subreddit-stratified train/val/test splits (60/20/20)
- Multi-hot label encoding for multi-label classification

## Model Architecture

- **Base Model**: BERT-base-uncased
- **Classification Head**: Linear layer with 27 outputs
- **Loss Function**: BCEWithLogitsLoss for multi-label learning
- **Threshold Optimization**: Per-emotion F1 score maximization
- **Evaluation**: Micro/macro precision, recall, F1-score

## Results Directory Structure
```
results/
├── model_checkpoints/
├── downsampling_results.csv
├── underperforming_labels.txt
└── augmentation_experiment_results.csv
```

For detailed usage examples, see `examples.py`.

