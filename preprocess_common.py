import pandas as pd

EMOTION_COLS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

def clean_unclear_examples(df):
    return df[df["example_very_unclear"] == False]

def group_by_id_and_labels(df):
    grouped = df.groupby("id").agg({
        "text": "first",
        "subreddit": "first",
        "created_utc": "first",
        **{col: "sum" for col in EMOTION_COLS}
    }).reset_index()

    for col in EMOTION_COLS:
        grouped[col] = (grouped[col] > 0).astype(int)

    grouped["num_labels"] = grouped[EMOTION_COLS].sum(axis=1)
    grouped = grouped[grouped["num_labels"] > 0]
    grouped.drop(columns=["num_labels"], inplace=True)

    return grouped[["id", "text", "subreddit", "created_utc"] + EMOTION_COLS]
