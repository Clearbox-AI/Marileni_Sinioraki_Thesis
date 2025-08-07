"""BERT-based model for multilabel emotion classification."""

import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the BERT model."""
    model_name: str = "bert-base-uncased"
    num_labels: int = 28  # GoEmotions has 28 emotion labels
    max_length: int = 512
    dropout_rate: float = 0.1
    hidden_size: Optional[int] = None
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    def __post_init__(self):
        if self.hidden_size is None:
            # Will be set based on the model's config
            pass


class BertForMultiLabelClassification(nn.Module):
    """BERT model for multilabel emotion classification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Get hidden size from BERT config
        if config.hidden_size is None:
            self.hidden_size = self.bert.config.hidden_size
        else:
            self.hidden_size = config.hidden_size
            
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier layer."""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Ground truth labels for training
            
        Returns:
            Dictionary containing logits and optionally loss
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        result = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            result["loss"] = loss
            
        return result
    
    def predict(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Make predictions with the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            threshold: Threshold for binary classification
            
        Returns:
            Binary predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).long()
            
        return predictions
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs["logits"]
            probabilities = torch.sigmoid(logits)
            
        return probabilities
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig = None):
        """
        Load a pre-trained model from a directory.
        
        Args:
            model_path: Path to the saved model
            config: Model configuration
            
        Returns:
            Loaded model
        """
        if config is None:
            config = ModelConfig()
            
        model = cls(config)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")
        
        # Save config
        import json
        config_dict = {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "max_length": self.config.max_length,
            "dropout_rate": self.config.dropout_rate,
            "hidden_size": self.hidden_size
        }
        
        with open(f"{save_directory}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)


class MultiLabelTrainer(Trainer):
    """Custom trainer for multilabel classification."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the training loss.
        
        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return outputs
            
        Returns:
            Loss and optionally outputs
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Use BCEWithLogitsLoss for multilabel classification
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss


def create_model(config: ModelConfig = None) -> BertForMultiLabelClassification:
    """
    Create a BERT model for multilabel classification.
    
    Args:
        config: Model configuration
        
    Returns:
        BERT model
    """
    if config is None:
        config = ModelConfig()
        
    return BertForMultiLabelClassification(config)


def get_model_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Get the tokenizer for the specified model.
    
    Args:
        model_name: Name of the pre-trained model
        
    Returns:
        Tokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)
