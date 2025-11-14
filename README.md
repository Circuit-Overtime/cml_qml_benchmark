# Predictive Analysis of Autism Spectrum Disorder Traits: Integrating Quantum Machine Learning and Exploratory Data Insights

This repository contains the code and resources for the research paper **"Predictive Analysis of Autism Spectrum Disorder Traits: Integrating Quantum Machine Learning and Exploratory Data Insights"**. The project compares classical and quantum machine learning (QML) models for predicting Autism Spectrum Disorder (ASD) traits using demographic and questionnaire-based data.

## Overview

- **Goal:** Evaluate and compare the effectiveness of classical ML and QML models for ASD trait prediction.
- **Dataset:** Publicly available ASD screening dataset (3,744 instances, 17 features) from Kaggle.
- **Models:** 
    - Classical: Logistic Regression, Naive Bayes, Decision Tree, KNN, SVM, Random Forest, Histogram Gradient Boosting Classifier (HGBC)
    - Quantum: Quantum Support Vector Machine (QSVM), Variational Quantum Classifier (VQC), Quantum Neural Network (QNN)
- **Key Finding:** HGBC achieved the highest accuracy (97.86%), while QML models showed potential but were limited by current hardware and dataset constraints.

## Features

- Data preprocessing (label encoding, scaling, PCA for QML)
- Extensive model benchmarking and hyperparameter tuning
- Confusion matrix, precision, recall, F1-score, and accuracy evaluation
- Exploratory Data Analysis (EDA) on demographic factors (age, sex, ethnicity)
- Comparative study with previous ASD prediction works

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `qiskit`, `pennylane`, etc.

### Installation

1. Clone the repository:
     ```bash
     git clone https://github.com/yourusername/asd-ml-qml.git
     cd asd-ml-qml
     ```
2. Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

### Usage

1. Download the ASD dataset from Kaggle and place it in the `data/` directory.
2. Run the main analysis script:
     ```bash
     python main.py
     ```
3. Results (metrics, plots) will be saved in the `results/` directory.

## Results

| Model         | Accuracy | Precision | Recall | F1-score |
|---------------|----------|-----------|--------|----------|
| HGBC          | 97.86%   | 98.48%    | 97.50% | 97.90%   |
| SVM           | 97.06%   | 96.32%    | 98.25% | 97.28%   |
| Random Forest | 95.99%   | 96.95%    | 95.50% | 96.22%   |
| QSVM (QML)    | 62.00%   | 65.00%    | 59.00% | 61.80%   |
| VQC (QML)     | 54.47%   | 61.35%    | 38.69% | 47.46%   |

*See the paper for full results and discussion.*

## Exploratory Data Insights

- ASD traits are most prevalent in children (60.57%) and males.
- Ethnicity and family history also influence ASD trait prevalence.

## Citation

If you use this work, please cite:

```
Kolay, S., Samaddar, S., Bhattacharya, A., Bhabani, B., Das, T., Siddique, S., & Maity, I. (2024). Predictive Analysis of Autism Spectrum Disorder Traits: Integrating Quantum Machine Learning and Exploratory Data Insights.
```

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, please contact [Ayushman Bhattacharya](mailto:bhattacharyaa599@gmail.com).
