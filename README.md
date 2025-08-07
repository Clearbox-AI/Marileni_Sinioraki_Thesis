# Multilabel Emotion Classification with BERT - Kaggle Kernel

This Kaggle kernel implements a complete pipeline for multilabel emotion classification on the GoEmotions dataset using BERT-based transformers.

## Overview

The pipeline includes:
- **Data Loading**: Automatic detection and loading of GoEmotions dataset
- **Preprocessing**: Text cleaning and tokenization with BERT tokenizer
- **Model**: BERT-based multilabel classifier with custom classification head
- **Training**: PyTorch training loop with validation
- **Evaluation**: Comprehensive multilabel metrics

## Quick Start

### Option 1: Run the Script
```bash
python main.py
```

### Option 2: Use the Notebook
Open `emotion_classification_notebook.ipynb` in Kaggle and run all cells.

## Configuration

The kernel is pre-configured for Kaggle environment:
- Input path: `/kaggle/input`
- Output path: `/kaggle/working`
- GPU enabled by default
- Optimized batch sizes and epochs for Kaggle time limits

You can modify the configuration in `config.json` or directly in the `Config` class in `main.py`.

## Expected Results

On the GoEmotions dataset, you can expect:
- **F1 Macro**: 0.45-0.55
- **F1 Micro**: 0.50-0.60
- **Accuracy**: 0.35-0.45
- **Hamming Loss**: 0.10-0.20

## Dataset

This kernel works with the GoEmotions dataset, which contains:
- 58,000+ Reddit comments
- 28 emotion labels (27 emotions + neutral)
- Multilabel format (comments can have multiple emotions)

Add the GoEmotions dataset to your Kaggle kernel from:
- Dataset: `google-research-datasets/go-emotions`

## Features

### Kaggle Optimizations
- Reduced training epochs (2) and max sequence length (128) for faster execution
- Automatic fallback to sample data if dataset not found
- GPU utilization with CUDA detection
- Memory-efficient batch processing

### Error Handling
- Graceful handling of missing datasets
- Fallback to sample data for demonstration
- Comprehensive error logging
- Robust package import handling

### Output
- Training progress logs
- Detailed evaluation metrics
- Model saving (if enabled)
- Results saved as JSON

## Technical Details

### Model Architecture
- Base: BERT (bert-base-uncased)
- Classification head: Linear layer with dropout
- Loss function: BCEWithLogitsLoss for multilabel classification
- Optimizer: AdamW with linear warmup

### Data Processing
- Tokenization: BERT tokenizer with max length 128
- Padding and truncation for batch processing
- Train/validation/test splits: 80/10/10

### Metrics
- F1 scores (macro, micro, weighted)
- Precision and recall (macro, micro)
- Accuracy (subset accuracy)
- Hamming loss
- Jaccard score

## Files Structure

```
├── main.py                           # Main execution script
├── emotion_classification_notebook.ipynb  # Jupyter notebook version
├── config.json                       # Configuration file
├── requirements.txt                  # Package dependencies
├── kernel-metadata.json             # Kaggle kernel metadata
└── README.md                        # This file
```

## Usage Examples

### Basic Usage
The kernel runs automatically when executed in Kaggle. No additional setup required.

### Custom Configuration
Modify the `Config` class in `main.py`:

```python
config = Config()
config.model_name = "distilbert-base-uncased"  # Use DistilBERT
config.batch_size = 32                         # Increase batch size
config.num_epochs = 3                          # More training epochs
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` in config
2. **Time Limit**: Reduce `num_epochs` or use DistilBERT
3. **Dataset Not Found**: The kernel creates sample data automatically
4. **Package Import Errors**: All required packages are pre-installed in Kaggle

### Performance Tips

1. Use GPU acceleration (enabled by default)
2. Adjust batch size based on available memory
3. Use DistilBERT for faster training
4. Reduce max_length for shorter sequences

## License

MIT License - Feel free to use and modify.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multilabel_emotion_classification_kaggle,
  author = {Sinioraki, Marileni},
  title = {Multilabel Emotion Classification with BERT - Kaggle Kernel},
  year = {2025},
  url = {https://kaggle.com/mariakalo/multilabel-emotion-classification}
}
```
