# Marileni_Sinioraki_Thesis

# GoEmotions Multi-Label Classification – Thesis Codebase

This repository contains the codebase for my thesis on **multi-label emotion classification** using the [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions). The core goal is to explore how different **data splitting strategies** affect generalization in emotion classification models.

---

## 🧠 Key Features

- Uses the full annotated GoEmotions dataset
- Cleans and filters ambiguous examples
- Aggregates annotations into multi-label vectors
- Implements multiple data splitting strategies:
  - `unseen_subreddits`: Generalization to subreddits not seen during training
  - `equally_splitted_subreddits`: Balanced distribution across train/val/test
  - `time_series`: Temporal split (old posts in train, new in test)
  - `single_label`: Filters to only examples with full annotator agreement

---

## 🗂️ Project Structure

