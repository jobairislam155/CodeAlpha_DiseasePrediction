# CodeAlpha_DiseasePrediction

Machine Learning Internship @ **CodeAlpha** - Task 4: *Disease Prediction from Medical Data*

## Problem
Predict whether a breast tumor is **malignant** or **benign** from 30 diagnostic measurements (radius, texture, concavity, etc.).

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository, loaded via `sklearn.datasets.load_breast_cancer`.
569 samples, 30 numeric features, no missing values. 357 benign / 212 malignant.

## Approach
1. EDA: class balance, correlation heatmap, feature distributions
2. Stratified 80/20 train/test split
3. `StandardScaler` inside a `Pipeline` (no data leakage)
4. Models: Logistic Regression, SVM, Random Forest, XGBoost, compared with 5-fold CV and a held-out test set
5. GridSearchCV tuning of SVM (best: linear kernel, C=0.1)
6. Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, ROC curves
7. Model saved with `joblib`, with an example prediction

## Results (test set, 114 samples)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| SVM (RBF) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |
| XGBoost | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9927 |
| Random Forest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 |
| **Tuned SVM (final)** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | 0.9937 |

Metrics treat "benign" as the positive class. For the malignant class, precision and recall are both 0.98.

## How to run
```bash
git clone https://github.com/<your-username>/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
pip install -r requirements.txt
jupyter notebook Disease_Prediction.ipynb
```

## Limitations
Small dataset, a single train/test split and no external validation. This is an educational project, not a medical diagnostic tool.
