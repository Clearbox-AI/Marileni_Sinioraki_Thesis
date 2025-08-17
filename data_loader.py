"""
Data loading utilities for GoEmotions multi-label emotion classification.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import os
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer


class GoEmotionsDataLoader:
    """Data loader for preprocessed GoEmotions dataset."""
    
    def __init__(self, data_path: str = "data/processed_goemotions.csv"):
        self.data_path = data_path
        self.emotion_cols = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness',
            'optimism', 'pride', 'realization', 'relief', 'remorse',
            'sadness', 'surprise'
        ]
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load the preprocessed dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please run prepare_data.py first.")
        
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded dataset with {len(self.df)} samples")
        return self.df
    
    def get_texts_and_labels(self) -> Tuple[List[str], np.ndarray]:
        """Extract texts and multi-label arrays."""
        if self.df is None:
            self.load_data()
        
        texts = self.df['text'].tolist()
        labels = self.df[self.emotion_cols].values.astype(np.float32)
        
        return texts, labels
    
    def get_label_statistics(self) -> dict:
        """Get statistics about label distribution."""
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_samples': len(self.df),
            'num_emotions': len(self.emotion_cols),
            'emotion_frequencies': self.df[self.emotion_cols].sum().to_dict(),
            'avg_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).mean(),
            'max_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).max(),
            'min_labels_per_sample': self.df[self.emotion_cols].sum(axis=1).min()
        }
        
        return stats
    
    def get_emotion_names(self) -> List[str]:
        """Get list of emotion names."""
        return self.emotion_cols
    
    def get_sample_by_id(self, sample_id: int) -> dict:
        """Get a specific sample by ID."""
        if self.df is None:
            self.load_data()
        
        sample = self.df[self.df['id'] == sample_id]
        if len(sample) == 0:
            return None
        
        sample = sample.iloc[0]
        return {
            'id': sample['id'],
            'text': sample['text'],
            'subreddit': sample['subreddit'],
            'emotions': [emotion for emotion in self.emotion_cols if sample[emotion] == 1]
        }
    
    def filter_by_emotions(self, emotions: List[str]) -> pd.DataFrame:
        """Filter dataset to include only samples with specific emotions."""
        if self.df is None:
            self.load_data()
        
        # Check if all emotions are valid
        invalid_emotions = [e for e in emotions if e not in self.emotion_cols]
        if invalid_emotions:
            raise ValueError(f"Invalid emotions: {invalid_emotions}")
        
        # Filter samples that have at least one of the specified emotions
        mask = self.df[emotions].sum(axis=1) > 0
        filtered_df = self.df[mask]
        
        print(f"Filtered dataset: {len(filtered_df)} samples with emotions {emotions}")
        return filtered_df
    
    def get_subreddit_statistics(self) -> dict:
        """Get statistics about subreddit distribution."""
        if self.df is None:
            self.load_data()
        
        subreddit_counts = self.df['subreddit'].value_counts()
        
        return {
            'unique_subreddits': self.df['subreddit'].nunique(),
            'most_common_subreddit': subreddit_counts.index[0],
            'most_common_count': subreddit_counts.iloc[0],
            'least_common_count': subreddit_counts.min(),
            'avg_posts_per_subreddit': subreddit_counts.mean()
        }
    
    def split_by_subreddit(self, train_ratio: float = 0.6, val_ratio: float = 0.2, 
                          test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by subreddit to ensure no data leakage.
        Each subreddit's data is split proportionally across train/val/test.
        """
        if self.df is None:
            self.load_data()
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        train_dfs, val_dfs, test_dfs = [], [], []
        
        print("Splitting dataset by subreddit...")
        
        for subreddit, group in self.df.groupby("subreddit"):
            # First split: separate test set
            train_val, test = train_test_split(
                group,
                test_size=test_ratio,
                random_state=42,
                shuffle=True
            )
            
            # Second split: divide remaining into train and validation
            val_size = val_ratio / (train_ratio + val_ratio)
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                random_state=42,
                shuffle=True
            )
            
            train_dfs.append(train)
            val_dfs.append(val)
            test_dfs.append(test)
        
        # Combine all splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_labels_for_hf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert emotion columns to HuggingFace format with 'labels' column."""
        df_copy = df.copy()
        df_copy["labels"] = df_copy[self.emotion_cols].values.tolist()
        return df_copy
    
    def convert_to_hf_dataset(self, df: pd.DataFrame) -> Dataset:
        """Convert pandas DataFrame to HuggingFace Dataset format."""
        # Prepare labels column
        df_with_labels = self.prepare_labels_for_hf(df)
        
        # Convert to HuggingFace Dataset
        hf_ds = Dataset.from_pandas(df_with_labels)
        
        # Remove individual emotion columns, keep 'labels' column
        hf_ds = hf_ds.map(
            lambda x: {"labels": [x[col] for col in self.emotion_cols]}, 
            remove_columns=self.emotion_cols
        )
        
        return hf_ds
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer_name: str = "bert-base-uncased", 
                        max_length: int = 128) -> Dataset:
        """
        Tokenize text data using specified tokenizer.
        
        Args:
            dataset: HuggingFace Dataset
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        tokenized_dataset.set_format(
            "torch", 
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        return tokenized_dataset
    
    def prepare_training_datasets(self, tokenizer_name: str = "bert-base-uncased",
                                max_length: int = 128, train_ratio: float = 0.6,
                                val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Complete pipeline to prepare tokenized datasets for training.
        
        Returns:
            Tuple of (train_ds, val_ds, test_ds) as tokenized HuggingFace Datasets
        """
        # Split data by subreddit
        train_df, val_df, test_df = self.split_by_subreddit(train_ratio, val_ratio, test_ratio)
        
        # Print emotion statistics for training set
        print("\nTraining set emotion frequencies:")
        emotion_counts = train_df[self.emotion_cols].sum().sort_values(ascending=False)
        for emotion, count in emotion_counts.items():
            print(f"{emotion}: {int(count)}")
        
        # Convert to HuggingFace datasets
        train_ds = self.convert_to_hf_dataset(train_df)
        val_ds = self.convert_to_hf_dataset(val_df)
        test_ds = self.convert_to_hf_dataset(test_df)
        
        # Tokenize datasets
        train_ds = self.tokenize_dataset(train_ds, tokenizer_name, max_length)
        val_ds = self.tokenize_dataset(val_ds, tokenizer_name, max_length)
        test_ds = self.tokenize_dataset(test_ds, tokenizer_name, max_length)
        
        print(f"\nDatasets prepared with tokenizer: {tokenizer_name}")
        print(f"Max length: {max_length}")
        
        return train_ds, val_ds, test_ds
    
    def get_raw_dataframes(self, train_ratio: float = 0.6,
                          val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Get raw DataFrames for train/val/test splits (not tokenized).
        Useful for downsampling experiments that need to modify the raw data.
        
        Returns:
            Tuple of (train_df, val_df, test_df) as pandas DataFrames
        """
        return self.split_by_subreddit(train_ratio, val_ratio, test_ratio)
    
    def downsample_by_label_reduction(self, df: pd.DataFrame,
                                    reduction_pct: float,
                                    random_state: int = 42) -> pd.DataFrame:
        """
        Downsample each emotion by a given percentage in a multi-label DataFrame.

        Args:
            df:            Original DataFrame containing emotion columns.
            reduction_pct: Percentage to remove from each label (0–100).
            random_state:  Seed for reproducibility.

        Returns:
            A new DataFrame with ~reduction_pct% of each emotion's examples removed,
            and prints before/after counts sorted by original frequency.
        """
        # 1. Compute original & target counts per label (Series indexed by emotion_cols)
        orig_counts = df[self.emotion_cols].sum().astype(int)
        target_counts = (orig_counts * (1 - reduction_pct / 100.0)).astype(int)

        # 2. Move to numpy for disambiguated indexing
        Y = df[self.emotion_cols].values.astype(int)             
        current_counts = orig_counts.values.copy()          
        target_vals    = target_counts.values               

        keep = np.ones(len(df), dtype=bool)
        rng = np.random.RandomState(random_state)

        # 3. Greedy removal
        for idx in rng.permutation(len(df)):
            labels = np.where(Y[idx] == 1)[0]
            if np.all(current_counts[labels] - 1 >= target_vals[labels]):
                keep[idx] = False
                current_counts[labels] -= 1
                if np.all(current_counts <= target_vals):
                    break

        # 4. Build downsampled DataFrame
        down_df = df.iloc[keep].reset_index(drop=True)

        # 5. Print summary sorted by original frequency, using .loc to avoid warnings
        new_counts = down_df[self.emotion_cols].sum().astype(int)
        sorted_emotions = sorted(self.emotion_cols, key=lambda e: orig_counts.loc[e], reverse=True)

        print(f"\nDownsampled by {reduction_pct:.1f}% per label:")
        print("Label         Orig → New  (% remaining)")
        for emo in sorted_emotions:
            o = orig_counts.loc[emo]
            n = new_counts.loc[emo]
            pct = n / o * 100
            print(f"{emo:12s}: {o:5d} → {n:5d}  ({pct:5.1f}%)")

        return down_df
    
    def eda_augment_labels(self, df: pd.DataFrame, 
                          target_labels: List[str],
                          target_count: int,
                          alpha_sr: float = 0.1,
                          random_state: int = 42) -> pd.DataFrame:
        """
        Apply EDA (Easy Data Augmentation) to specific labels using synonym replacement.
        
        Args:
            df: Original DataFrame
            target_labels: List of emotion labels to augment
            target_count: Target number of examples for each label
            alpha_sr: Proportion of words to replace with synonyms
            random_state: Seed for reproducibility
            
        Returns:
            DataFrame with EDA-augmented examples
        """
        print(f"\nEDA augmentation for labels: {target_labels}")
        print(f"Target count per label: {target_count}")
        
        # Set random seed
        random.seed(random_state)
        np.random.seed(random_state)
        
        synthetic_rows = []
        
        for label in target_labels:
            if label not in self.emotion_cols:
                print(f"Warning: {label} not found in emotion columns, skipping...")
                continue
            
            # Get existing examples for this label
            label_mask = df[label] == 1
            label_samples = df[label_mask]
            existing_count = len(label_samples)
            
            if existing_count == 0:
                print(f"Warning: No samples found for {label}, skipping...")
                continue
            
            needed_count = target_count - existing_count
            if needed_count <= 0:
                print(f"No augmentation needed for {label} (already has {existing_count} examples)")
                continue
            
            print(f"  {label:12s}: {existing_count:5d} → {target_count:5d} (+{needed_count:5d})")
            
            # Create real templates for this label
            real_templates = label_samples["text"].tolist()
            
            # Generate synthetic examples
            for i in range(needed_count):
                base_text = random.choice(real_templates)
                aug_text = self._eda_synonym_replacement(base_text, alpha_sr)
                
                # Create label vector (only this emotion is active)
                label_vector = {e: int(e == label) for e in self.emotion_cols}
                
                synthetic_row = {
                    "id": f"eda_{label}_{i + 10000}",
                    "text": aug_text,
                    "subreddit": "synthetic",
                    "created_utc": int(datetime.utcnow().timestamp()),
                    **label_vector
                }
                synthetic_rows.append(synthetic_row)
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        print(f"\nGenerated {len(synthetic_df)} synthetic examples using EDA")
        
        return synthetic_df
    
    def _eda_synonym_replacement(self, sentence: str, alpha_sr: float = 0.1) -> str:
        """Apply synonym replacement for EDA augmentation."""
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download required NLTK data if not present
            try:
                wordnet.synsets('test')
            except LookupError:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
        except ImportError:
            print("Warning: NLTK not available, returning original sentence")
            return sentence
        
        words = sentence.split()
        n = len(words)
        n_sr = max(1, int(alpha_sr * n))
        
        # Apply synonym replacement
        words_copy = words.copy()
        for _ in range(n_sr):
            counter = 0
            synonyms = []
            while counter < 10 and not synonyms:
                word = random.choice(words_copy)
                synonyms = self._get_synonyms(word)
                counter += 1
            if synonyms:
                idx = words_copy.index(word)
                words_copy[idx] = random.choice(synonyms)
        
        return " ".join(words_copy)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        try:
            from nltk.corpus import wordnet
            
            syns = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != word.lower():
                        syns.add(name)
            return list(syns)
        except:
            return []
    
    def llm_augment_labels(self, df: pd.DataFrame,
                          target_labels: List[str], 
                          target_count: int,
                          llm_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                          batch_size: int = 8,
                          random_state: int = 42) -> pd.DataFrame:
        """
        Apply LLM-based augmentation to specific labels.
        
        Args:
            df: Original DataFrame
            target_labels: List of emotion labels to augment
            target_count: Target number of examples for each label
            llm_model_id: HuggingFace model ID for LLM
            batch_size: Batch size for LLM generation
            random_state: Seed for reproducibility
            
        Returns:
            DataFrame with LLM-generated examples
        """
        print(f"\nLLM augmentation for labels: {target_labels}")
        print(f"Target count per label: {target_count}")
        print(f"Using model: {llm_model_id}")
        
        try:
            import torch
            import re
            from tqdm import tqdm
            from transformers import AutoModelForCausalLM, pipeline
        except ImportError:
            print("Error: Required packages for LLM augmentation not available")
            return pd.DataFrame()
        
        # Set random seed
        random.seed(random_state)
        torch.manual_seed(random_state)
        
        # Load LLM
        print("Loading LLM model...")
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, use_fast=False)
            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            llm_pipe = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_tokenizer,
                pad_token_id=llm_model.config.eos_token_id
            )
        except Exception as e:
            print(f"Error loading LLM model: {e}")
            return pd.DataFrame()
        
        # System prompt
        system_prompt = """You are a helpful assistant that rewrites sentences for social media.
You receive a sentence and an emotion. You have to rewrite the sentence preserving the original meaning and emotion.
Do not add hashtags, emojis, or any new content unless already present.
Keep the same text style.
Return only one sentence.
Examples:
Original: I just finished my first half marathon.
Rewritten: Just completed my first half marathon!
Original: So grateful to all my colleagues for pulling this off. 🙏
Rewritten: Huge thanks to my colleagues for making this happen. 🙏
Original: Wrapping up a great quarter with an amazing team.
Rewritten: Finishing a fantastic quarter alongside a great team."""
        
        synthetic_rows = []
        
        for label in target_labels:
            if label not in self.emotion_cols:
                print(f"Warning: {label} not found in emotion columns, skipping...")
                continue
            
            # Get existing examples for this label
            label_mask = df[label] == 1
            label_samples = df[label_mask]
            existing_count = len(label_samples)
            
            if existing_count == 0:
                print(f"Warning: No samples found for {label}, skipping...")
                continue
            
            needed_count = target_count - existing_count
            if needed_count <= 0:
                print(f"No augmentation needed for {label} (already has {existing_count} examples)")
                continue
            
            print(f"  Generating {needed_count} examples for {label}...")
            
            # Create real templates for this label
            real_templates = label_samples["text"].tolist()
            originals = random.choices(real_templates, k=needed_count)
            
            # Prepare prompts
            prompts = []
            for orig in originals:
                chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Sentence: '{orig}'\nEmotion: {label}\nRewrite:"}
                ]
                prompt = llm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
            
            # Generate in batches
            for b in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {label}"):
                batch_prompts = prompts[b: b + batch_size]
                try:
                    results = llm_pipe(
                        batch_prompts,
                        max_new_tokens=40,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95
                    )
                    
                    for i, result in enumerate(results):
                        full_text = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
                        # Remove prompt
                        gen_only = full_text.replace(batch_prompts[i], "", 1).strip()
                        cleaned = re.split(r'[\""\n]', gen_only.strip())[0].strip()
                        
                        if not cleaned or len(cleaned.split()) < 3:
                            continue
                        
                        # Create label vector (only this emotion is active)
                        label_vector = {e: int(e == label) for e in self.emotion_cols}
                        
                        synthetic_row = {
                            "id": f"llm_{label}_{b+i + 10000}",
                            "text": cleaned,
                            "original_text": originals[b + i],
                            "subreddit": "synthetic_llm", 
                            "created_utc": int(datetime.utcnow().timestamp()),
                            **label_vector
                        }
                        
                        synthetic_rows.append(synthetic_row)
                        
                except Exception as e:
                    print(f"Error in batch {b}: {e}")
                    continue
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        print(f"\nGenerated {len(synthetic_df)} synthetic examples using LLM")
        
        return synthetic_df