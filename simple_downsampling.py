"""
Simple downsampling functionality for GoEmotions dataset.
Contains only the essential code needed for downsampling experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TrainingArguments, BertForSequenceClassification
from datasets import Dataset
from model import MultilabelTrainer


def downsample_by_label_reduction(df: pd.DataFrame,
                                  emotion_cols: list[str],
                                  reduction_pct: float,
                                  random_state: int = 42) -> pd.DataFrame:
    """
    Downsample each emotion by a given percentage in a multi-label DataFrame.

    Args:
        df:            Original DataFrame containing emotion columns.
        emotion_cols:  List of column names (one per emotion).
        reduction_pct: Percentage to remove from each label (0–100).
        random_state:  Seed for reproducibility.

    Returns:
        A new DataFrame with ~reduction_pct% of each emotion's examples removed.
    """
    # 1. Compute original & target counts per label
    orig_counts = df[emotion_cols].sum().astype(int)
    target_counts = (orig_counts * (1 - reduction_pct / 100.0)).astype(int)

    # 2. Move to numpy for efficient processing
    Y = df[emotion_cols].values.astype(int)             
    current_counts = orig_counts.values.copy()          
    target_vals = target_counts.values               

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

    # 5. Print summary sorted by original frequency
    new_counts = down_df[emotion_cols].sum().astype(int)
    sorted_emotions = sorted(emotion_cols, key=lambda e: orig_counts.loc[e], reverse=True)

    print(f"\nDownsampled by {reduction_pct:.1f}% per label:")
    print("Label         Orig → New  (% remaining)")
    for emo in sorted_emotions:
        o = orig_counts.loc[emo]
        n = new_counts.loc[emo]
        pct = n / o * 100
        print(f"{emo:12s}: {o:5d} → {n:5d}  ({pct:5.1f}%)")

    return down_df


def convert_to_hf(df, emotion_cols):
    """Convert DataFrame to HuggingFace dataset."""
    labels = df[emotion_cols].values.astype(float)
    dataset_dict = {
        'text': df['text'].tolist(),
        'labels': labels.tolist()
    }
    return Dataset.from_dict(dataset_dict)


def tokenize(examples, tokenizer):
    """Tokenize examples."""
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )


def run_downsampling_experiments(train_df, val_df, test_df, emotion_cols, 
                               tune_thresholds_func, compute_metrics_func,
                               print_class_report_func):
    """
    Run downsampling experiments exactly as in your original code.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        emotion_cols: List of emotion column names
        tune_thresholds_func: Function to tune thresholds
        compute_metrics_func: Function to compute metrics
        print_class_report_func: Function to print classification report
    """
    reductions = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    metrics_list = []

    for pct in reductions:
        print(f"\n{'='*60}")
        print(f"REDUCTION: {pct}%")
        print('='*60)
        
        # 1. Downsample the train DataFrame
        down_train = downsample_by_label_reduction(
            train_df, emotion_cols, reduction_pct=pct, random_state=123
        )

        # 2. Prepare datasets
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        train_ds = convert_to_hf(down_train, emotion_cols)
        val_ds = convert_to_hf(val_df, emotion_cols)
        test_ds = convert_to_hf(test_df, emotion_cols)
        
        # Tokenize
        train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
        val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
        test_ds = test_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
        
        # Set format for PyTorch
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # 3. Initialize fresh BERT model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(emotion_cols),
            problem_type="multi_label_classification"
        )

        # 4. Set up training
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=8,
            weight_decay=0.01,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_micro/f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to="wandb",
            run_name="goemotions-multilabel-bert"
        )
        
        # 5. Create trainer
        trainer_wrapper = MultilabelTrainer()
        trainer = trainer_wrapper.create_trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics_fn=compute_metrics_func
        )
        
        # 6. Train the model
        trainer.train()

        print("Reduction: ", pct)

        # 7. Tune thresholds on validation set
        eval_output = trainer.predict(val_ds)
        val_logits = eval_output.predictions
        val_labels = eval_output.label_ids
        thresholds = tune_thresholds_func(val_logits, val_labels)
        
        val_metrics = compute_metrics_func((val_logits, val_labels), thresholds)
        print("Validation metrics:")
        print(val_metrics)
        print_class_report_func(val_logits, val_labels, emotion_cols, thresholds)
        
        # 8. Evaluate on test set
        test_output = trainer.predict(test_ds)
        test_logits = test_output.predictions
        test_labels = test_output.label_ids
        
        test_metrics = compute_metrics_func((test_logits, test_labels), thresholds)
        print("Test set metrics:")
        print(test_metrics)
        print_class_report_func(test_logits, test_labels, emotion_cols, thresholds)
        
        # 9. Save metrics
        test_metrics["reduction"] = pct
        metrics_list.append(test_metrics)

    # 10. Create results DataFrame
    metrics_df = pd.DataFrame(metrics_list).set_index("reduction")

    # 11. Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df.index, metrics_df["micro/f1"], marker="o", label="micro/F1")
    plt.plot(metrics_df.index, metrics_df["macro/f1"], marker="s", label="macro/F1")
    plt.xlabel("Downsampling Reduction (%)")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 vs. Training Set Downsampling")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return metrics_df
