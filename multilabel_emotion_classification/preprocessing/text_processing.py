"""
Text preprocessing utilities.
"""

from typing import Dict, List, Optional, Union
import logging

# Import with fallback handling
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextTokenizer:
    """Handles text tokenization for model training."""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Name of the pretrained model/tokenizer
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info(f"Initialized tokenizer: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize tokenizer: {e}")
        else:
            logger.error("Transformers library not available")
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenize text examples.
        
        Args:
            examples: Dictionary with 'text' key
            
        Returns:
            Dictionary with tokenized inputs
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        
        return self.tokenizer(
            examples["text"],
            padding="max_length", 
            truncation=True,
            max_length=self.max_length
        )
    
    def tokenize_datasets(self, train_ds: 'Dataset', val_ds: 'Dataset', test_ds: 'Dataset') -> tuple:
        """
        Tokenize all datasets and set format for PyTorch.
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            test_ds: Test dataset
            
        Returns:
            Tuple of tokenized datasets
        """
        if not DATASETS_AVAILABLE:
            raise ImportError("datasets library not available")
        
        logger.info("Tokenizing datasets...")
        
        # Apply tokenization
        train_ds = train_ds.map(self.tokenize_function, batched=True)
        val_ds = val_ds.map(self.tokenize_function, batched=True)
        test_ds = test_ds.map(self.tokenize_function, batched=True)
        
        # Set format for PyTorch
        columns = ["input_ids", "attention_mask", "labels"]
        train_ds.set_format("torch", columns=columns)
        val_ds.set_format("torch", columns=columns)
        test_ds.set_format("torch", columns=columns)
        
        logger.info("Dataset tokenization completed")
        return train_ds, val_ds, test_ds


def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = text.strip()
    
    # Add any additional preprocessing steps here
    # e.g., removing special characters, normalization, etc.
    
    return text
