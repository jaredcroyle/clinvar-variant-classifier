# 🧬 Genomic Variant Classifier Pipeline

This project implements an **end-to-end pipeline** for classifying genomic variants as **Benign (0)** or **Pathogenic (1)** using **Random Forest** and **XGBoost** models trained on **ClinVar** data.  
Data source: [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)

---

## Overview

- **Data**: Variant summary from ClinVar (`data/variant_summary.txt.gz`)
- **Preprocessing**: Data cleaning and feature selection
- **Models**: 
  - Random Forest (sklearn)
  - XGBoost (xgboost)
- **Training**: Grid search with cross-validation for hyperparameter tuning
- **Evaluation**: Accuracy, ROC AUC, Precision, Recall, F1-score
- **Visualization**: Confusion matrix, ROC curve, Precision-Recall curve, Decision tree plot (Random Forest), SHAP plots (XGBoost)
- **Reports**: HTML report summarizing metrics and figures

---

## How to Run

```bash
python main.py

Project Structure

├── data/
│   └── variant_summary.txt.gz    # Input data (TSV)
├── src/
│   ├── data_preprocessing.py     # Data loading & cleaning (specify dataset size in "int=")
│   ├── model_training.py         # Model training & tuning
│   ├── model_evaluation.py       # Evaluation & plotting
│   └── shap_visualization.py     # SHAP plot generation
├── utils/
│   ├── report_generator.py       # Plot/report saving
│   └── model_utils.py            # Model saving/loading
├── report/                       # Generated reports and figures
├── main.py                       # Pipeline entry point
└── report_template.html          # HTML report template