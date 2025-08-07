# Multilabel Emotion Classification

A modular implementation for training and evaluating multilabel emotion classification models on the GoEmotions dataset using BERT-based transformers.

## Overview

This package provides a complete pipeline for multilabel emotion classification, including:

- **Data Loading & Preprocessing**: Clean and prepare the GoEmotions dataset
- **Text Processing**: Advanced tokenization and preprocessing utilities
- **Data Augmentation**: EDA (Easy Data Augmentation) and LLM-based augmentation
- **Model Architecture**: BERT-based models for multilabel classification
- **Training Pipeline**: Custom trainer with early stopping and evaluation
- **Evaluation Metrics**: Comprehensive multilabel metrics and threshold tuning
- **Configuration Management**: Flexible configuration system
- **Utilities**: Logging, experiment tracking, and visualization tools

## Installation

### From Source

```bash
git clone https://github.com/marileni/multilabel-emotion-classification.git
cd multilabel-emotion-classification
pip install -e .
```

### Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from multilabel_emotion_classification import Config
from multilabel_emotion_classification.main import main

# Load default configuration
config = Config.get_default()

# Run training and evaluation
main()
```

### Command Line Interface

```bash
# Basic training
emotion-classify --data-path ./data --output-dir ./results

# Custom configuration
emotion-classify --config config.json --experiment-name my_experiment

# Evaluation only
emotion-classify --eval-only --model-path ./model --data-path ./data

# With augmentation
emotion-classify --use-eda --use-llm-aug --data-path ./data
```

### Configuration

Create a configuration file:

```python
from multilabel_emotion_classification.config import Config

# Get default config
config = Config.get_default()

# Modify as needed
config.model.model_name = "distilbert-base-uncased"
config.training.learning_rate = 3e-5
config.training.batch_size = 32

# Save configuration
config.to_json("my_config.json")
```

## Package Structure

```
multilabel_emotion_classification/
├── __init__.py
├── main.py                    # Main script and CLI
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── config.py             # Configuration classes
│   └── settings.py           # Default settings and constants
├── data/                     # Data loading and splitting
│   ├── __init__.py
│   └── loader.py            # GoEmotions data loader
├── preprocessing/            # Text processing and augmentation
│   ├── __init__.py
│   ├── text_processing.py   # Tokenization and preprocessing
│   └── augmentation.py      # Data augmentation techniques
├── models/                  # Model definitions
│   ├── __init__.py
│   └── bert_model.py       # BERT-based multilabel classifier
├── training/               # Training pipeline
│   ├── __init__.py
│   ├── trainer.py         # Custom trainer class
│   └── utils.py           # Training utilities
├── evaluation/            # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py        # Multilabel metrics computation
│   ├── threshold_tuning.py # Threshold optimization
│   └── reports.py        # Report generation and visualization
└── utils.py              # General utilities
```

## Features

### Data Processing

- **GoEmotions Dataset**: Support for loading and processing the GoEmotions dataset
- **Data Cleaning**: Remove duplicates, filter by length, handle missing values
- **Data Splitting**: Configurable train/validation/test splits
- **Label Processing**: Handle multilabel format and emotion mappings

### Text Preprocessing

- **Tokenization**: BERT-compatible tokenization with proper handling
- **Text Cleaning**: URL removal, mention handling, whitespace normalization
- **Length Management**: Truncation and padding to specified lengths

### Data Augmentation

- **EDA (Easy Data Augmentation)**:
  - Synonym replacement
  - Random insertion
  - Random swap
  - Random deletion

- **LLM-based Augmentation**:
  - GPT-based text generation
  - Emotion-aware paraphrasing
  - Contextual variations

### Model Architecture

- **BERT-based Models**: Support for various BERT variants
- **Multilabel Head**: Custom classification head for multilabel prediction
- **Dropout and Regularization**: Configurable dropout for better generalization
- **Model Serialization**: Save and load trained models

### Training Pipeline

- **Custom Trainer**: Extends HuggingFace Trainer for multilabel classification
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Learning Rate Scheduling**: Linear warmup and decay
- **Gradient Clipping**: Stability during training
- **Mixed Precision**: FP16 support for faster training

### Evaluation

- **Comprehensive Metrics**:
  - F1 scores (macro, micro, weighted, samples)
  - Precision and recall (macro, micro, weighted, samples)
  - Accuracy, Hamming loss, Jaccard score
  - Exact match ratio
  - Per-class metrics

- **Threshold Tuning**:
  - Global threshold optimization
  - Per-class threshold optimization
  - Multiple optimization metrics

- **Visualization**:
  - Confusion matrices
  - Performance comparison plots
  - Training curves

### Configuration Management

- **Flexible Config System**: Dataclass-based configuration
- **JSON Support**: Load and save configurations as JSON
- **Command Line Override**: Override config values from command line
- **Validation**: Automatic configuration validation

## Usage Examples

### Basic Training

```python
from multilabel_emotion_classification import Config
from multilabel_emotion_classification.data import GoEmotionsDataLoader
from multilabel_emotion_classification.models import BertForMultiLabelClassification
from multilabel_emotion_classification.training import setup_training

# Load configuration
config = Config.get_default()
config.data.data_path = "./data"
config.model.model_name = "bert-base-uncased"
config.training.num_epochs = 3

# Load data
data_loader = GoEmotionsDataLoader(config.data.data_path)
df = data_loader.load_data()

# Train model (simplified)
# ... (see main.py for complete example)
```

### Custom Evaluation

```python
from multilabel_emotion_classification.evaluation import (
    compute_multilabel_metrics,
    find_optimal_thresholds,
    generate_classification_report
)

# Compute metrics
metrics = compute_multilabel_metrics(
    y_true=y_true,
    y_pred=y_pred,
    label_names=emotion_labels
)

# Find optimal thresholds
thresholds = find_optimal_thresholds(
    y_true=y_true,
    y_prob=y_prob,
    metric='f1_macro'
)

# Generate report
report = generate_classification_report(
    metrics=metrics,
    label_names=emotion_labels
)
print(report)
```

### Data Augmentation

```python
from multilabel_emotion_classification.preprocessing import EDAugmenter

# Initialize augmenter
augmenter = EDAugmenter(
    alpha_sr=0.1,  # Synonym replacement
    alpha_ri=0.1,  # Random insertion
    alpha_rs=0.1,  # Random swap
    alpha_rd=0.1,  # Random deletion
    num_aug=4
)

# Augment texts
augmented_samples = augmenter.augment_batch(
    texts=["I love this movie!", "This is terrible!"],
    labels=[[1, 0, 0], [0, 1, 0]]
)
```

## Configuration Options

### Data Configuration

```python
data_config = {
    "data_path": "./data",
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "min_length": 10,
    "max_length": 512,
    "remove_duplicates": True
}
```

### Model Configuration

```python
model_config = {
    "model_name": "bert-base-uncased",
    "num_labels": 28,
    "max_length": 512,
    "dropout_rate": 0.1
}
```

### Training Configuration

```python
training_config = {
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "early_stopping": True,
    "early_stopping_patience": 3
}
```

## GoEmotions Dataset

This package is designed to work with the GoEmotions dataset, which contains:

- **27 emotion categories + neutral**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

- **Multilabel format**: Each text can have multiple emotion labels

- **Reddit comments**: Real-world conversational text data

## Performance

The package achieves competitive performance on the GoEmotions dataset:

- **F1 Macro**: ~0.55-0.65 (depending on model and configuration)
- **F1 Micro**: ~0.60-0.70
- **Accuracy**: ~0.45-0.55
- **Training time**: ~2-4 hours on GPU for full dataset

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multilabel_emotion_classification,
  author = {Sinioraki, Marileni},
  title = {Multilabel Emotion Classification with BERT},
  year = {2024},
  url = {https://github.com/marileni/multilabel-emotion-classification}
}
```

## Acknowledgments

- GoEmotions dataset by Google Research
- HuggingFace Transformers library
- PyTorch framework
- scikit-learn for evaluation metrics
