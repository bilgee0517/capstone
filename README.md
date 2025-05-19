# Cross-Lingual Post-Training Techniques for Low-Resource Languages

**Author:** Bilegjargal Altangerel

## Project Overview

This repository contains the implementation and analysis of cross-lingual post-training techniques for improving language model performance on low-resource languages, with a specific focus on Mongolian. The project explores various parameter-efficient fine-tuning methods to enhance model adaptability while minimizing computational requirements.

## Key Features

- Implementation of multiple fine-tuning techniques:
- Comprehensive evaluation framework
- Data processing and cleaning pipelines
- Perplexity-based performance analysis
- Visualization tools for results analysis

## Repository Structure

```
.
├── evaluation/               # Evaluation scripts and results
│   ├── results/             # Evaluation results
│   ├── mm-eval/            # Multilingual evaluation tools
│   ├── visualizations.ipynb # Results visualization
│   ├── qwen_evaluation.py   # Qwen model evaluation
│   └── flores_evaluation.py # FLORES benchmark evaluation
├── utils/                   # Training and utility functions
│   ├── finetune_qwen.py    # Qwen model fine-tuning
│   ├── finetune_nllb.py    # NLLB model fine-tuning
│   ├── train_lora.py       # LoRA training implementation
│   ├── train_embed.py      # Embedding training
│   ├── train_tokenizer.py  # Tokenizer training
│   ├── continued_training.py # Continued pre-training
│   └── translate_mongolian.py # Mongolian translation utilities
├── data_cleaning.ipynb      # Data preprocessing notebook
├── working.ipynb           # Experimentation notebook
├── perplexity.py           # Perplexity calculation script
├── requirements.txt        # Project dependencies
└── Paper.pdf              # Project documentation
```

## Requirements

The project requires Python 3.8+ and the following key dependencies:
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- NLTK
- Pandas
- NumPy

For a complete list of dependencies, see `requirements.txt`.

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Run the data cleaning notebook:
```bash
jupyter notebook data_cleaning.ipynb
```

### Model Training

The training scripts are located in the `utils/` directory. Each script serves a specific purpose:

1. Fine-tune Qwen model:
```bash
python utils/finetune_qwen.py
```

2. Fine-tune NLLB model:
```bash
python utils/finetune_nllb.py
```

3. Train with LoRA:
```bash
python utils/train_lora.py
```

4. Train embeddings:
```bash
python utils/train_embed.py
```

5. Train tokenizer:
```bash
python utils/train_tokenizer.py
```

6. Continue pre-training:
```bash
python utils/continued_training.py
```

### Evaluation

1. Run perplexity evaluation:
```bash
python perplexity.py
```

2. For comprehensive evaluation:
```bash
python evaluation/qwen_evaluation.py
python evaluation/flores_evaluation.py
```

## Results

The project evaluates model performance using:
- Perplexity scores
- FLORES benchmark metrics
- Custom evaluation metrics for Mongolian language tasks

Detailed results and visualizations can be found in the `evaluation/results/` directory.

## Citation

If you use this code in your research, please cite:
```
[Citation information from Paper.pdf]
```

## License

[Specify license information]

## Contact

For questions and feedback, please contact:
- Bilegjargal Altangerel
- [Contact information]

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
  - [Data Collection](#data-collection)
  - [Data Cleaning](#data-cleaning)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contact](#contact)

## Introduction

The rapid development of natural language processing (NLP) has led to significant advancements in language modeling. However, low-resource languages often lag due to limited data availability and computational resources. This project addresses these challenges by exploring cross-lingual post-training techniques that enhance the performance of pre-trained language models on low-resource languages, with a focus on Mongolian.

## Methodology

### Data Collection

- **Sources:** Collected Mongolian text data from various sources, including news articles, literature, and public domain texts.
- **Scope:** Focused on diverse topics to ensure a comprehensive language representation.

### Data Cleaning

- **Tools Used:** Utilized Jupyter notebooks (`data_cleaning.ipynb`) for data preprocessing.
- **Processes:** Removed duplicates, handled missing values, and normalized text to improve data quality.

### Model Training

- **Techniques Implemented:**
  - **Qwen Fine-tuning:** Implementation of fine-tuning for the Qwen model (`utils/finetune_qwen.py`)
  - **NLLB Fine-tuning:** Fine-tuning implementation for the NLLB model (`utils/finetune_nllb.py`)
  - **LoRA Training:** Parameter-efficient fine-tuning using LoRA (`utils/train_lora.py`)
  - **Embedding Training:** Custom embedding training (`utils/train_embed.py`)
  - **Tokenizer Training:** Mongolian-specific tokenizer training (`utils/train_tokenizer.py`)
  - **Continued Pre-training:** Additional pre-training steps (`utils/continued_training.py`)
- **Implementation:** All training scripts are located in the `utils/` directory, with detailed configurations and training processes.

### Evaluation

- **Metrics:** Evaluated model performance using perplexity scores.
- **Scripts:** 
  - `perplexity.py` for perplexity calculation
  - `evaluation/qwen_evaluation.py` for Qwen model evaluation
  - `evaluation/flores_evaluation.py` for FLORES benchmark evaluation

## Results

The experiments demonstrated varying degrees of improvement across the different fine-tuning techniques. Detailed results, including performance metrics and comparative analyses, are documented within the repository.

## Conclusion

The findings underscore the potential of parameter-efficient fine-tuning strategies in enhancing the performance of language models for low-resource languages. The comparative analysis provides insights into the trade-offs and benefits of each technique, guiding future research and application.

## Repository Structure

