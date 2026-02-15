# Credit Card Fraud Detection

This GitHub Repository implements a credit card fraud detection system. It focuses on exploring a highly imbalanced dataset, preprocessing the data, and training various machine learning models to identify fraudulent transactions. The project extends an existing solution by introducing new models (Naive Bayes and a Multi-Layer Perceptron) and comparing their performance against the original models.

## Dataset

The dataset used for this project is the `Credit Card Fraud Detection` dataset, imported from KaggleHub (`mlg-ulb/creditcardfraud`). This dataset contains anonymized transaction data with 28 principal components (V1-V28), `Time`, `Amount`, and a `Class` label (0 for non-fraudulent, 1 for fraudulent).

## Models Used

The notebook evaluates the following classification models:

### New Models (Fayez & Issa Extension):
*   **Naive Bayes (GaussianNB)**: Trained on both imbalanced and balanced datasets.
*   **Multi-Layer Perceptron (MLP) Neural Network**: Implemented using PyTorch, trained on the imbalanced dataset with Focal Loss.

### Original Models (Janio Martinez Bachmann):
*   **Logistic Regression**
*   **K-Nearest Neighbors (KNeighborsClassifier)**
*   **Support Vector Classifier (SVC)**
*   **Decision Tree Classifier**

These models are evaluated after various preprocessing steps, including robust scaling and different approaches to handle class imbalance (undersampling, SMOTE through `imblearn` pipelines).

## Key Steps

1.  **Data Import and Initial Exploration**: Loading the dataset and examining its basic statistics and class distribution (which is highly imbalanced).
2.  **Data Preprocessing**: 
    *   Standardizing `Amount` and `Time` features using `RobustScaler`.
    *   Splitting the data into training and testing sets.
3.  **New Model Training & Evaluation (Part 2)**:
    *   Training Naive Bayes on both the original imbalanced data and a randomly undersampled balanced dataset.
    *   Training a PyTorch-based MLP with Focal Loss on the original imbalanced data.
    *   Generating classification reports and confusion matrices for these models.
4.  **Data Balancing (Undersampling)**: Creating a new undersampled dataset for the original models by randomly sampling non-fraudulent transactions to match the number of fraudulent transactions.
5.  **Outlier Removal**: Identifying and removing outliers from the undersampled dataset for features `V14`, `V12`, and `V10` using the Interquartile Range (IQR) method.
6.  **Original Model Training & Evaluation (Part 3)**:
    *   Training Logistic Regression, K-Nearest Neighbors, SVC, and Decision Tree Classifier on the undersampled data.
    *   Using `GridSearchCV` to find optimal hyperparameters for these classifiers.
    *   Performing cross-validation and calculating ROC AUC scores.
    *   Evaluating these models on both the undersampled test set and the original imbalanced test set, providing classification reports, confusion matrices, and ROC curves.

## How to Run

1.  **Open in Google Colab**: Upload or open this notebook directly in Google Colab.
2.  **Run All Cells**: Execute all cells in the notebook sequentially. The notebook will download the dataset from KaggleHub and proceed with all data preprocessing, model training, and evaluation steps.
3.  **Review Outputs**: Observe the printed classification reports, confusion matrices, and generated plots for each model's performance.
