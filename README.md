# Cross-Lingual Post-Training Techniques for Low-Resource Languages

## Project Title: Efficiency of Cross-Lingual Post-Training Techniques
Author: Bilegjargal Altangerel

## Project Overview
This repository contains the code, datasets, analysis, and documentation for the Capstone Project titled Efficiency of Cross-Lingual Post-Training Techniques. The project investigates the efficiency of cross-lingual post-training techniques aimed at improving the linguistic adaptability and performance of pre-trained language models, specifically for low-resource languages like Mongolian.

Through a comparative analysis of existing methodologies such as Adapter layers, BitFit, and prompt tuning, the project aims to enhance data efficiency and accuracy for multilingual models while reducing resource consumption. This research focuses on parameter-efficient fine-tuning strategies that optimize performance across languages with minimal computational overhead.

# Table of Contents
Project Overview
Repository Structure
Datasets
Approach
Installation
Usage
Contributing
License
References


``` bash 

├── data/
│   ├── mn.txt/        # Mongolian dataset from CC-100
│   └── processed/              # Preprocessed datasets
├── src/
│   ├── model_training/         # Code for training and fine-tuning models
│   ├── post_training_methods/  # Implementations of post-training techniques (e.g., Adapter, BitFit)
│   ├── evaluation/             # Evaluation scripts and metrics (e.g., BLEU, accuracy, perplexity)
│   └── utils/                  # Utility scripts for data processing and embedding analysis
├── results/
│   ├── experimental_data/      # Results of experiments
│   └── visualizations/         # Graphs and plots for result visualization
├── docs/
│   ├── paper/                  # Research paper draft and write-ups
│   └── slides/                 # Project presentation slides
├── README.md                   # Project overview
└── requirements.txt            # Dependencies required for running the project

```

## Datasets
This project uses the Mongolian text dataset sourced from the CC-100 dataset. The dataset comprises 397M words, and it has been preprocessed for tokenization, sentence segmentation, and embedding analysis.

 - Raw data: Located in data/cc100-mongolian/.
 - Processed data: Available in data/processed/ for ready-to-use experiments.

### Dataset Preprocessing
The data was cleaned and tokenized using methods to remove noise (such as punctuation, URLs, and numbers). 


## Approach

### Key Focus
The project aims to optimize and compare the following post-training techniques:

- Adapter Layers: Lightweight modules added between transformer layers.
- BitFit: A fine-tuning approach that updates only the bias terms.
- Prompt Tuning: Fine-tuning a small number of additional tokens prepended to the input sequences.
### The project involves:

1. Comparative Analysis: Testing different post-training techniques on the CC-100 Mongolian dataset.
2. Experimental Design: Evaluating model performance through metrics like BLEU score, perplexity, and accuracy.
3. Embedding Adaptation: Exploring strategies to incorporate Mongolian language embeddings into pre-trained LLMs without retraining the entire embedding layer.
### Techniques Implemented:
- Adapter layers
- BitFit
- Prompt Tuning
Evaluation Metrics: BLEU, perplexity, accuracy
