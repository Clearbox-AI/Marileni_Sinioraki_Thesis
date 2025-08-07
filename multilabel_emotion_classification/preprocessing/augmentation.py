"""
Data preprocessing and augmentation utilities.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging

# Import with fallback handling
try:
    from nltk.corpus import wordnet
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataDownsampler:
    """Handles downsampling of training data to balance label distribution."""
    
    @staticmethod
    def downsample_by_labels(df: pd.DataFrame, 
                           emotion_cols: List[str],
                           fraction: float = 0.20,
                           random_state: int = 42) -> pd.DataFrame:
        """
        Downsample data by retaining a fraction of examples for each label.
        
        Args:
            df: Input DataFrame
            emotion_cols: List of emotion column names
            fraction: Fraction of examples to retain for each label
            random_state: Random seed
            
        Returns:
            Downsampled DataFrame
        """
        logger.info(f"Downsampling data with fraction {fraction}")
        
        selected_indices = set()
        
        # Process each label independently
        for label in emotion_cols:
            label_rows = df[df[label] == 1]
            n_keep = int(len(label_rows) * fraction)
            
            if n_keep > 0:
                sampled = label_rows.sample(n=n_keep, random_state=random_state)
                selected_indices.update(sampled.index)
        
        # Combine selected rows (deduplicated)
        downsampled_df = df.loc[list(selected_indices)].reset_index(drop=True)
        
        logger.info(f"Original size: {len(df)}, Downsampled size: {len(downsampled_df)}")
        return downsampled_df


class EDAugmenter:
    """Easy Data Augmentation (EDA) techniques for text augmentation."""
    
    def __init__(self):
        self.wordnet_available = WORDNET_AVAILABLE
        if not self.wordnet_available:
            logger.warning("NLTK WordNet not available. EDA will use limited functionality.")
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        if not self.wordnet_available:
            return []
        
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    syns.add(name)
        return list(syns)
    
    def random_deletion(self, words: List[str], p: float) -> List[str]:
        """Randomly delete words with probability p."""
        if len(words) == 1:
            return words
        new_words = [w for w in words if random.random() > p]
        return new_words if new_words else [random.choice(words)]
    
    def random_swap(self, words: List[str], n_swaps: int) -> List[str]:
        """Randomly swap positions of words."""
        new_words = words.copy()
        for _ in range(n_swaps):
            if len(new_words) >= 2:
                i, j = random.sample(range(len(new_words)), 2)
                new_words[i], new_words[j] = new_words[j], new_words[i]
        return new_words
    
    def random_insertion(self, words: List[str], n_insert: int) -> List[str]:
        """Randomly insert synonyms."""
        new_words = words.copy()
        for _ in range(n_insert):
            counter = 0
            synonyms = []
            while counter < 10 and not synonyms:
                if new_words:
                    candidate = random.choice(new_words)
                    synonyms = self.get_synonyms(candidate)
                counter += 1
            if synonyms:
                new_words.insert(random.randint(0, len(new_words)), random.choice(synonyms))
        return new_words
    
    def synonym_replacement(self, words: List[str], n_replace: int) -> List[str]:
        """Replace words with synonyms."""
        new_words = words.copy()
        for _ in range(n_replace):
            counter = 0
            synonyms = []
            while counter < 10 and not synonyms and new_words:
                candidate = random.choice(new_words)
                synonyms = self.get_synonyms(candidate)
                counter += 1
            if synonyms and candidate in new_words:
                idx = new_words.index(candidate)
                new_words[idx] = random.choice(synonyms)
        return new_words
    
    def eda(self, sentence: str, 
            alpha_sr: float = 0.1, 
            alpha_ri: float = 0.0, 
            alpha_rs: float = 0.0, 
            p_rd: float = 0.0) -> str:
        """
        Apply EDA transformations to a sentence.
        
        Args:
            sentence: Input sentence
            alpha_sr: Synonym replacement rate
            alpha_ri: Random insertion rate
            alpha_rs: Random swap rate
            p_rd: Random deletion probability
            
        Returns:
            Augmented sentence
        """
        words = sentence.split()
        n = len(words)
        
        # Calculate number of operations
        n_sr = max(1, int(alpha_sr * n))
        n_ri = max(1, int(alpha_ri * n))
        n_rs = max(1, int(alpha_rs * n))
        
        # Apply transformations
        words = self.synonym_replacement(words, n_sr)
        words = self.random_insertion(words, n_ri)
        words = self.random_swap(words, n_rs)
        words = self.random_deletion(words, p_rd)
        
        return " ".join(words)


class LLMBasedAugmenter:
    """LLM-based text generation for data augmentation."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", device: Union[str, int] = "auto"):
        """
        Initialize LLM-based augmenter.
        
        Args:
            model_name: Name of the text generation model
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        self.generator = None
        self.transformers_available = TRANSFORMERS_AVAILABLE
        
        if self.transformers_available:
            try:
                self.generator = pipeline(
                    "text2text-generation", 
                    model=model_name, 
                    device=device if device != "auto" else -1
                )
                logger.info(f"Initialized LLM augmenter with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM augmenter: {e}")
                self.generator = None
        else:
            logger.warning("Transformers library not available. LLM augmentation disabled.")
    
    def generate_sentences(self, emotion: str, n: int = 5) -> List[str]:
        """
        Generate sentences expressing a given emotion.
        
        Args:
            emotion: Target emotion
            n: Number of sentences to generate
            
        Returns:
            List of generated sentences
        """
        if not self.generator:
            logger.warning("LLM generator not available")
            return []
        
        prompt_templates = [
            "Write a short sentence expressing the emotion '{}' without using the word.",
            "Give an example of how someone might express '{}' implicitly.",
            "How would someone express the emotion '{}' without naming it?",
            "Create a realistic sentence that conveys '{}'."
        ]
        
        sentences = []
        for _ in range(n):
            prompt = random.choice(prompt_templates).format(emotion)
            try:
                output = self.generator(
                    prompt, 
                    max_new_tokens=30, 
                    temperature=0.9, 
                    do_sample=True
                )
                text = output[0]['generated_text'].strip()
                
                # Filter out sentences that include the emotion word or are too short
                if emotion.lower() not in text.lower() and len(text.split()) >= 4:
                    sentences.append(text)
            except Exception as e:
                logger.error(f"Error generating sentence for {emotion}: {e}")
                continue
        
        return sentences


class SyntheticDataGenerator:
    """Generates synthetic data for rare emotion labels."""
    
    def __init__(self, emotion_cols: List[str]):
        """
        Initialize synthetic data generator.
        
        Args:
            emotion_cols: List of all emotion column names
        """
        self.emotion_cols = emotion_cols
        self.eda_augmenter = EDAugmenter()
        self.llm_augmenter = None
    
    def initialize_llm_augmenter(self, model_name: str = "google/flan-t5-base", device: Union[str, int] = "auto"):
        """Initialize LLM-based augmenter."""
        self.llm_augmenter = LLMBasedAugmenter(model_name, device)
    
    def extract_real_templates(self, df: pd.DataFrame, 
                             target_labels: List[str], 
                             examples_per_label: int = 10) -> Dict[str, List[str]]:
        """
        Extract real text examples for each target label.
        
        Args:
            df: Source DataFrame
            target_labels: Labels to extract examples for
            examples_per_label: Number of examples per label
            
        Returns:
            Dictionary mapping labels to example texts
        """
        real_templates = {}
        
        for label in target_labels:
            label_examples = df[df[label] == 1]['text'].drop_duplicates()
            
            # Sample examples (take all if fewer than requested)
            n_samples = min(examples_per_label, len(label_examples))
            if n_samples > 0:
                real_templates[label] = label_examples.sample(
                    n=n_samples, 
                    random_state=42
                ).tolist()
            else:
                real_templates[label] = []
                logger.warning(f"No examples found for label: {label}")
        
        return real_templates
    
    def generate_eda_synthetic_data(self, real_templates: Dict[str, List[str]], 
                                  samples_per_label: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data using EDA techniques.
        
        Args:
            real_templates: Dictionary mapping labels to example texts
            samples_per_label: Number of synthetic samples per label
            
        Returns:
            DataFrame with synthetic samples
        """
        logger.info(f"Generating EDA synthetic data with {samples_per_label} samples per label")
        
        synthetic_rows = []
        
        for label in real_templates:
            if not real_templates[label]:
                logger.warning(f"No templates available for label: {label}")
                continue
                
            for i in range(samples_per_label):
                base_text = random.choice(real_templates[label])
                augmented_text = self.eda_augmenter.eda(base_text)
                
                # Create multi-hot label vector
                row = {
                    "id": f"eda_{label}_{i}",
                    "text": augmented_text,
                    "subreddit": "synthetic",
                    "created_utc": int(datetime.utcnow().timestamp()),
                    **{col: int(col == label) for col in self.emotion_cols}
                }
                
                synthetic_rows.append(row)
        
        df_synthetic = pd.DataFrame(synthetic_rows)
        logger.info(f"Generated {len(df_synthetic)} EDA synthetic samples")
        return df_synthetic
    
    def generate_llm_synthetic_data(self, target_labels: List[str], 
                                  samples_per_label: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data using LLM-based generation.
        
        Args:
            target_labels: Labels to generate data for
            samples_per_label: Number of synthetic samples per label
            
        Returns:
            DataFrame with synthetic samples
        """
        if not self.llm_augmenter or not self.llm_augmenter.generator:
            logger.error("LLM augmenter not initialized")
            return pd.DataFrame()
        
        logger.info(f"Generating LLM synthetic data with {samples_per_label} samples per label")
        
        synthetic_rows = []
        
        for label in target_labels:
            logger.info(f"Generating samples for label: {label}")
            
            # Generate texts in batches
            batch_size = 50
            generated_texts = []
            
            for batch_start in range(0, samples_per_label, batch_size):
                batch_end = min(batch_start + batch_size, samples_per_label)
                batch_size_actual = batch_end - batch_start
                
                batch_texts = self.llm_augmenter.generate_sentences(label, batch_size_actual)
                generated_texts.extend(batch_texts)
            
            # Create rows for generated texts
            for i, text in enumerate(generated_texts):
                if i >= samples_per_label:
                    break
                    
                row = {
                    "id": f"llm_{label}_{i}",
                    "text": text,
                    "subreddit": "synthetic", 
                    "created_utc": int(datetime.utcnow().timestamp()),
                    **{col: int(col == label) for col in self.emotion_cols}
                }
                
                synthetic_rows.append(row)
        
        df_synthetic = pd.DataFrame(synthetic_rows)
        logger.info(f"Generated {len(df_synthetic)} LLM synthetic samples")
        return df_synthetic
    
    def augment_dataset(self, train_df: pd.DataFrame, 
                       least_supported_labels: List[str],
                       method: str = "eda",
                       samples_per_label: int = 1000,
                       examples_per_label: int = 10) -> pd.DataFrame:
        """
        Augment training dataset with synthetic data.
        
        Args:
            train_df: Original training DataFrame
            least_supported_labels: Labels that need augmentation
            method: Augmentation method ("eda" or "llm")
            samples_per_label: Number of synthetic samples per label
            examples_per_label: Number of real examples to use as templates (for EDA)
            
        Returns:
            Augmented DataFrame combining original and synthetic data
        """
        logger.info(f"Augmenting dataset using {method} method")
        
        if method == "eda":
            real_templates = self.extract_real_templates(
                train_df, least_supported_labels, examples_per_label
            )
            synthetic_df = self.generate_eda_synthetic_data(real_templates, samples_per_label)
        elif method == "llm":
            synthetic_df = self.generate_llm_synthetic_data(least_supported_labels, samples_per_label)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
        
        # Combine original and synthetic data
        augmented_df = pd.concat([train_df, synthetic_df], ignore_index=True)
        
        logger.info(f"Augmented dataset size: {len(augmented_df)} (original: {len(train_df)}, synthetic: {len(synthetic_df)})")
        return augmented_df
