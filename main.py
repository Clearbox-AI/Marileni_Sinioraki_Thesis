import argparse
import sys
import os

# Add current directory to Python path for Kaggle compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# For Kaggle environment - add both working and src directories
kaggle_paths = ['/kaggle/working', '/kaggle/src']
for path in kaggle_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import modules
try:
    # Debug: check if files exist
    import os
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Files in /kaggle/src: {os.listdir('/kaggle/src') if os.path.exists('/kaggle/src') else 'Directory not found'}")
    print(f"Files in /kaggle/working: {os.listdir('/kaggle/working') if os.path.exists('/kaggle/working') else 'Directory not found'}")
    print(f"Python path: {sys.path}")
    
    from data_loader import download_data, load_raw_data
    from preprocess_common import clean_unclear_examples, group_by_id_and_labels
    from utils import print_emotion_counts, print_subreddit_stats, plot_emotion_distribution
    from preprocess_common import EMOTION_COLS
    
    # Strategy imports - these might be empty, so we'll handle them gracefully
    try:
        from unseen_subreddits import preprocess as unseen_preprocess
    except ImportError:
        print("Warning: unseen_subreddits.preprocess not found")
        unseen_preprocess = None
        
    try:
        from equally_splitted_subreddits import preprocess as equal_preprocess
    except ImportError:
        print("Warning: equally_splitted_subreddits.preprocess not found")
        equal_preprocess = None
        
    try:
        from single_label import preprocess as single_label_preprocess
    except ImportError:
        print("Warning: single_label.preprocess not found")
        single_label_preprocess = None
        
    try:
        from time_series import preprocess as time_preprocess
    except ImportError:
        print("Warning: time_series.preprocess not found")
        time_preprocess = None
        
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    
    # Fallback: define essential functions inline
    print("Falling back to inline function definitions...")
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    
    # Essential functions from data_loader
    def download_data(save_dir="data/full_dataset"):
        os.makedirs(save_dir, exist_ok=True)
        DATA_URLS = [
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
            "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
        ]
        for url in DATA_URLS:
            filename = os.path.join(save_dir, url.split("/")[-1])
            if not os.path.exists(filename):
                os.system(f"wget -P {save_dir} {url}")
    
    def load_raw_data(path="data/full_dataset"):
        df1 = pd.read_csv(os.path.join(path, "goemotions_1.csv"))
        df2 = pd.read_csv(os.path.join(path, "goemotions_2.csv"))
        df3 = pd.read_csv(os.path.join(path, "goemotions_3.csv"))
        return pd.concat([df1, df2, df3], ignore_index=True)
    
    # Functions from preprocess_common
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
    
    # Functions from utils
    def print_emotion_counts(df, emotion_cols):
        counts = df[emotion_cols].sum().sort_values(ascending=False)
        for emotion, count in counts.items():
            print(f"{emotion}: {int(count)}")
    
    def plot_emotion_distribution(df, emotion_cols):
        df[emotion_cols].sum().sort_values(ascending=False).plot(
            kind='bar', figsize=(12, 4), title="Emotion Frequency"
        )
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def print_subreddit_stats(df):
        print(f"Unique subreddits: {df['subreddit'].nunique()}")
        print(df['subreddit'].value_counts().head(10))

        subreddit_counts = df['subreddit'].value_counts()
        min_posts = subreddit_counts.min()
        print(f"Minimum posts in any subreddit: {min_posts}")
        print(f"Number of subreddits with that: {(subreddit_counts == min_posts).sum()}")
        print("Least common subreddits:")
        print(subreddit_counts[subreddit_counts == min_posts])
    
    # Define emotion columns
    EMOTION_COLS = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval',
        'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
        'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
        'gratitude', 'grief', 'joy', 'love', 'nervousness',
        'optimism', 'pride', 'realization', 'relief', 'remorse',
        'sadness', 'surprise', 'neutral'
    ]
    
    # Set strategy modules to None
    unseen_preprocess = None
    equal_preprocess = None
    single_label_preprocess = None
    time_preprocess = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline for GoEmotions thesis")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["unseen", "equal", "single", "time", "all"],
        default="all",
        help="Which preprocessing strategy to run"
    )
    return parser.parse_args()


def prepare_base_dataset():
    print("Downloading and loading dataset...")
    download_data()
    df = load_raw_data()

    print("Cleaning and aggregating...")
    df = clean_unclear_examples(df)
    df = group_by_id_and_labels(df)

    print("Final dataset length:", len(df))
    print_emotion_counts(df, EMOTION_COLS)
    print_subreddit_stats(df)
    plot_emotion_distribution(df, EMOTION_COLS)

    return df


def main():
    args = parse_args()
    df = prepare_base_dataset()

    if args.strategy == "unseen" or args.strategy == "all":
        print("\n=== Running Unseen Subreddits Split ===")
        if unseen_preprocess and hasattr(unseen_preprocess, 'run'):
            unseen_preprocess.run(df.copy())
        else:
            print("Unseen subreddits preprocessing not available (module not found or empty)")

    if args.strategy == "equal" or args.strategy == "all":
        print("\n=== Running Equally Split Subreddits ===")
        if equal_preprocess and hasattr(equal_preprocess, 'run'):
            equal_preprocess.run(df.copy())
        else:
            print("Equally split subreddits preprocessing not available (module not found or empty)")

    if args.strategy == "single" or args.strategy == "all":
        print("\n=== Running Single Label Consensus Only ===")
        if single_label_preprocess and hasattr(single_label_preprocess, 'run'):
            single_label_preprocess.run(df.copy())
        else:
            print("Single label preprocessing not available (module not found or empty)")

    if args.strategy == "time" or args.strategy == "all":
        print("\n=== Running Time Series Split ===")
        if time_preprocess and hasattr(time_preprocess, 'run'):
            time_preprocess.run(df.copy())
        else:
            print("Time series preprocessing not available (module not found or empty)")


if __name__ == "__main__":
    main()
