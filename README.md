# Predictive Analysis of Autism Spectrum Disorder Traits

## Integrating Quantum Machine Learning and Exploratory Data Insights

This project explores whether machine learning -- both classical and quantum -- can help identify Autism Spectrum Disorder (ASD) traits early, using simple demographic and questionnaire data instead of expensive neuroimaging procedures like MRI.

## What This Project Does

Autism Spectrum Disorder affects how people communicate, interact, and process information. Early detection is key to timely support, but current diagnostic methods are slow, costly, and often subjective. This research investigates a faster, data-driven alternative.

We trained and compared **10 different models** across three categories:

- **Classical ML** -- Standard algorithms (Logistic Regression, Naive Bayes, Decision Tree, KNN, SVM, Random Forest, Histogram Gradient Boosting) applied directly to the full feature set.
- **PCA-Reduced Classical ML** -- The same classical algorithms, but operating on a compressed version of the data using Principal Component Analysis.
- **Quantum ML** -- Quantum computing-based models (Variational Quantum Classifier, Quantum SVM, Quantum Naive Bayes) that encode data into quantum states for classification.

## Key Findings

| Approach | Best Model | Accuracy |
|----------|-----------|----------|
| Classical ML | Histogram Gradient Boosting (HGBC) | **97.86%** |
| Quantum ML | Quantum Naive Bayes (Q-NB) | 93.24% |
| Quantum ML | QSVM | 82.00% |

- **HGBC** achieved the strongest results overall: 97.86% accuracy, 98.48% precision, 97.50% recall, and 97.90% F1-score.
- **Quantum models** showed competitive performance, especially Q-NB, though they were constrained by simulation limits and small circuit sizes.
- **Exploratory analysis** revealed that ASD traits in this dataset were most prevalent in children (60.57% of cases), more common in males, and influenced by ethnicity and family history.

## Dataset

The study uses an augmented public dataset of **3,744 individuals** with 17 features, including 9 binary clinical screening questions (A1--A9) and demographic attributes (age, sex, ethnicity, family history). The target variable indicates the presence or absence of ASD traits.

## Repository Structure

```
├── data/              Dataset used for training and evaluation
├── paper/             LaTeX source and figures for the research paper
│   ├── figures/       All figures referenced in the paper
│   └── *.tex          Paper source file
├── scripts/           Python scripts (converted from notebooks)
│   ├── pca_classical.py     Classical ML with PCA reduction
│   ├── pca_quantum.py       Quantum ML models (VQC, QSVM)
│   └── visualization.py     Data visualization and EDA
└── README.md
```

## Running the Code

**Requirements:** Python 3.8+, scikit-learn, pandas, numpy, matplotlib, seaborn, PennyLane, Qiskit

```bash
pip install scikit-learn pandas numpy matplotlib seaborn pennylane qiskit qiskit-aer qiskit-machine-learning
```

Run the classical models:
```bash
python scripts/pca_classical.py
```

Run the quantum models:
```bash
python scripts/pca_quantum.py
```

Generate visualizations:
```bash
python scripts/visualization.py
```

## Authors

- Subhra Kolay -- JIS University, Kolkata
- Srijan Samaddar -- Netaji Subhash Engineering College, West Bengal
- Ayushman Bhattacharya -- JIS University, Kolkata
- Bidisha Bhabani -- JIS University, Kolkata
- Tanaya Das -- JIS University, Kolkata
- Sohrab Siddique -- JIS University, Kolkata
- Ilora Maity -- University of Luxembourg

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{kolay2025asd,
  title     = {Predictive Analysis of Autism Spectrum Disorder Traits: Integrating Quantum Machine Learning and Exploratory Data Insights},
  author    = {Kolay, Subhra and Samaddar, Srijan and Bhattacharya, Ayushman and Bhabani, Bidisha and Das, Tanaya and Siddique, Sohrab and Maity, Ilora},
  year      = {2025}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or collaborations, please contact [Ayushman Bhattacharya](mailto:bhattacharyaa599@gmail.com).
