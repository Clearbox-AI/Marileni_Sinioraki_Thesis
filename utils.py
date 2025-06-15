import matplotlib.pyplot as plt

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
