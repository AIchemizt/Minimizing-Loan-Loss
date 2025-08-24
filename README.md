# Minimizing Loan Loss: A Credit Card Default Prediction Model

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20CatBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project is a comprehensive machine learning solution designed to predict credit card default risk for a financial institution. Using a dataset of over 25,000 customer records, I developed a complete pipeline—from data cleaning and exploratory analysis to advanced modeling and threshold optimization. The final model serves as a powerful, interpretable tool to help the bank proactively identify high-risk customers and mitigate financial losses.

## Business Goal

The primary objective is to build a robust binary classification model that accurately predicts whether a credit card customer will default on their next month's payment. The model is optimized for the **F2 score**, a metric that heavily prioritizes **Recall**. This aligns with the business goal of minimizing False Negatives (failing to identify a defaulter), as the cost of a missed default is significantly higher than the cost of incorrectly flagging a non-defaulter.

## Tech Stack

- **Data Analysis & Manipulation:** `Pandas`, `NumPy`
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Machine Learning:** `Scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`
- **Development Environment:** `Jupyter Notebook`

## Workflow

1.  **Data Cleaning & EDA:** Handled inconsistencies and missing values in the dataset. Performed in-depth exploratory data analysis to visualize distributions and identify key relationships between customer attributes and default risk.
2.  **Feature Engineering:** Created new, financially meaningful features to better capture customer behavior, including delinquency streaks, credit utilization ratios, and payment-to-bill ratios over a 6-month period.
3.  **Model Training & Comparison:** Trained and evaluated several models, including Logistic Regression and several Gradient Boosting algorithms, using class weighting to handle the imbalanced nature of the dataset.
4.  **Hyperparameter Tuning:** Optimized the champion model (CatBoost) using `RandomizedSearchCV` to systematically search for the best combination of parameters, with the F2 score as the target metric.
5.  **Threshold Optimization:** Analyzed the precision-recall trade-off to determine the optimal classification threshold, ensuring the model's predictions were perfectly aligned with the project's F2-score maximization goal.

## Key Results

- **Best Model:** A tuned **CatBoost Classifier** was selected for its superior performance and interpretability.
- **Performance:** The final model achieved a maximum **F2 Score of 0.6027** on the hold-out test set.
- **Optimal Threshold:** The optimal probability threshold was identified as **0.4217**, balancing high recall with acceptable precision.

### Key Finding: The Power of Payment History

The most significant predictor of future default was the customer's most recent payment status (`PAY_0`). The analysis revealed a sharp, exponential increase in default risk as payment delays grew, confirming that recent delinquency is the most critical factor in credit risk assessment.


*Caption: Default rate skyrockets as payment delays increase, making it the most important predictive feature.*

## Repository Structure

```
.
├── Main.ipynb
├── README.md
├── requirements.txt
├── submission_catboost_final.csv
├── train_and_predict.py
├── train_dataset_final1.csv
└── validate_dataset_final.csv
```

## How to Run

To reproduce this project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AIchemizt/Minimizing-Loan-Loss.git
    cd Minimizing-Loan-Loss
    ```
2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Jupyter notebook** `Main.ipynb` to see the full analysis and modeling process.

---
*This project was completed as part of the Finance Club, IIT Roorkee's Open Project Summer 2025 program.*.