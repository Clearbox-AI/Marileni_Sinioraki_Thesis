import os
import pandas as pd

DATA_URLS = [
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv"
]

def download_data(save_dir="data/full_dataset"):
    os.makedirs(save_dir, exist_ok=True)
    for url in DATA_URLS:
        filename = os.path.join(save_dir, url.split("/")[-1])
        if not os.path.exists(filename):
            os.system(f"wget -P {save_dir} {url}")

def load_raw_data(path="data/full_dataset"):
    df1 = pd.read_csv(os.path.join(path, "goemotions_1.csv"))
    df2 = pd.read_csv(os.path.join(path, "goemotions_2.csv"))
    df3 = pd.read_csv(os.path.join(path, "goemotions_3.csv"))
    return pd.concat([df1, df2, df3], ignore_index=True)
