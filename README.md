# Marileni_Sinioraki_Thesis - GoEmotions Multi-Label Classification

This repository contains the codebase for the thesis on **multi-label emotion classification** using the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions).

---

## 🧠 Key Features

- Uses the full annotated GoEmotions dataset
- Implements multiple data splitting strategies:
  - `unseen_subreddits`: Generalization to subreddits not seen during training
  - `equally_splitted_subreddits`: Balanced distribution across train/val/test
  - `time_series`: Temporal split (old posts in train, new in test)
  - `single_label`: Filters to only examples with full annotator agreement

---

## 🗂️ Project Structure
│
├── unseen_subreddits/
│ ├── init.py
│ └── preprocess.py
├── equally_splitted_subreddits/
├── single_label/
├── time_series/
│ └── (same structure)
│
├── data_loader.py # Download & load raw dataset
├── preprocess_common.py # Dataset cleaning, label aggregation
├── utils.py # Stats, plotting, printing tools
├── main.py # CLI runner for all preprocessing pipelines
├── README.md
└── synthetic_data_tools_comparison.ipynb # Optional comparisons

## 🚀 Usage

Preprocessing is controlled via a command-line interface in main.py.

# Run all preprocessing strategies
python main.py

# Run only a specific strategy
python main.py --strategy unseen
python main.py --strategy equal
python main.py --strategy time
python main.py --strategy single
