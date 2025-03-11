# Cross-Lingual Post-Training Techniques for Low-Resource Languages

**Author:** Bilegjargal Altangerel

## Project Overview

This repository contains the code, datasets, analysis, and documentation for the Capstone Project titled "Efficiency of Cross-Lingual Post-Training Techniques." The project investigates the efficiency of cross-lingual post-training techniques aimed at improving the linguistic adaptability and performance of pre-trained language models, specifically for low-resource languages like Mongolian.

Through a comparative analysis of existing methodologies such as Adapter layers, BitFit, and prompt tuning, the project aims to enhance data efficiency and accuracy for multilingual models while reducing resource consumption. This research focuses on parameter-efficient fine-tuning strategies that optimize performance across languages with minimal computational overhead.

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

- **Techniques Compared:**
  - **Adapter Layers:** Integrated adapter modules into the pre-trained model to facilitate language adaptation.
  - **BitFit:** Applied bias-only fine-tuning to adjust specific model parameters.
  - **Prompt Tuning:** Employed prompt-based methods to guide the model's language understanding.
- **Implementation:** Conducted experiments using the `working.ipynb` notebook, detailing the training processes and configurations.

### Evaluation

- **Metrics:** Evaluated model performance using perplexity scores.
- **Script:** Used `perplexity.py` to calculate and analyze perplexity on the test datasets.

## Results

The experiments demonstrated varying degrees of improvement across the different fine-tuning techniques. Detailed results, including performance metrics and comparative analyses, are documented within the repository.

## Conclusion

The findings underscore the potential of parameter-efficient fine-tuning strategies in enhancing the performance of language models for low-resource languages. The comparative analysis provides insights into the trade-offs and benefits of each technique, guiding future research and application.

## Repository Structure

