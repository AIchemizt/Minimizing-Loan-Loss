"""
Credit Card Default Prediction: Full Training and Prediction Pipeline

This script encapsulates the entire workflow for the credit card default prediction project.
It performs the following steps:
1.  Loads the training and validation data.
2.  Applies a comprehensive preprocessing and feature engineering pipeline.
3.  Trains the final, optimized CatBoost classifier with the best hyperparameters.
4.  Determines the optimal classification threshold to maximize the F2 score.
5.  Generates the final submission file with predictions on the validation data.

To run this script:
`python train_and_predict.py`
"""

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, fbeta_score, precision_recall_curve
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")
print("Libraries imported successfully!")


# --- Step 2: Define Core Functions ---

def preprocess_data(df, is_train=True, scaler=None, train_cols=None):
    """
    Applies all cleaning and feature engineering steps to the dataframe.
    
    Args:
        df (pd.DataFrame): The input dataframe (train or validation).
        is_train (bool): Flag to indicate if this is the training set.
        scaler (StandardScaler): Fitted scaler from training data.
        train_cols (list): List of column names from the training data for alignment.
        
    Returns:
        pd.DataFrame: The processed dataframe.
        (if is_train) StandardScaler: The scaler fitted on the training data.
        (if is_train) list: The columns of the processed training data.
    """
    print(f"Starting preprocessing. Initial shape: {df.shape}")
    
    # --- Data Cleaning ---
    df.columns = df.columns.str.lower()
    
    # Impute missing 'age' with median
    if 'age' in df.columns and df['age'].isnull().any():
        # In a real scenario, the median would be saved from the training set.
        # For this script, we assume the provided training data is the source of truth.
        age_median = 29.0 # Hardcoded from notebook's df.describe() for consistency
        df['age'].fillna(age_median, inplace=True)

    # Clean categorical features
    df['education'] = df['education'].replace([0, 5, 6], 4)
    df['marriage'] = df['marriage'].replace(0, 3)
    
    # Robustly handle negative values by capping at 0, not removing rows
    if 'pay_to_bill_ratio' in df.columns:
        df['pay_to_bill_ratio'] = np.where(df['pay_to_bill_ratio'] < 0, 0, df['pay_to_bill_ratio'])
    if 'avg_bill_amt' in df.columns:
        df['avg_bill_amt'] = np.where(df['avg_bill_amt'] < 0, 0, df['avg_bill_amt'])

    # --- Feature Engineering ---
    pay_status_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    bill_amt_cols = ['bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6']
    pay_amt_cols = ['pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6']

    df['delinquency_count'] = (df[pay_status_cols] > 0).sum(axis=1)
    df['max_delinquency'] = df[pay_status_cols].max(axis=1)
    df['avg_utilization'] = df[bill_amt_cols].sum(axis=1) / (df['limit_bal'] * 6 + 1e-6)
    total_bill = df[bill_amt_cols].sum(axis=1)
    total_payment = df[pay_amt_cols].sum(axis=1)
    df['payment_to_bill_ratio_6m'] = total_payment / (total_bill + 1e-6)

    # --- One-Hot Encoding ---
    categorical_features = ['sex', 'education', 'marriage']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype='float32')

    if is_train:
        train_cols = df.drop(columns=['next_month_default', 'customer_id'], errors='ignore').columns.tolist()

    # --- Align Columns ---
    # Ensure validation set has the same columns as the training set
    if not is_train and train_cols is not None:
        df = df.reindex(columns=train_cols, fill_value=0)
    
    # --- Scaling ---
    numerical_cols = [
        'limit_bal', 'age', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
        'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
        'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
        'avg_bill_amt', 'pay_to_bill_ratio', 'delinquency_count', 'max_delinquency',
        'avg_utilization', 'payment_to_bill_ratio_6m'
    ]
    numerical_cols_exist = [col for col in numerical_cols if col in df.columns]

    if is_train:
        scaler = StandardScaler()
        df[numerical_cols_exist] = scaler.fit_transform(df[numerical_cols_exist])
        print(f"Preprocessing complete for training data. Final shape: {df.shape}")
        return df, scaler, train_cols
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for validation data.")
        df[numerical_cols_exist] = scaler.transform(df[numerical_cols_exist])
        print(f"Preprocessing complete for validation data. Final shape: {df.shape}")
        return df


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Trains the CatBoost model and evaluates it to find the optimal threshold.
    """
    print("\n--- Phase 4 & 5: Training Model and Optimizing Threshold ---")
    
    # Calculate class weight for the model
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Use the best parameters found during RandomizedSearchCV
    best_params = {
        'learning_rate': 0.05,
        'l2_leaf_reg': 5,
        'iterations': 1000,
        'depth': 4,
        'bagging_temperature': 0
    }
    
    # Initialize and train the final model
    final_model = CatBoostClassifier(
        class_weights=[1, scale_pos_weight],
        random_state=42,
        verbose=0,
        **best_params
    )
    
    print("Training final CatBoost model...")
    final_model.fit(X_train, y_train)
    
    # --- Threshold Optimization ---
    print("Optimizing classification threshold for F2 score...")
    y_probs = final_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    
    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx]
    max_f2_score = f2_scores[optimal_idx]
    
    print(f"\nOptimal Threshold found: {optimal_threshold:.4f}")
    print(f"Max F2 Score on Test Set: {max_f2_score:.4f}")
    
    # Show final classification report
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    print("\n--- Final Model Performance on Test Set (with Optimal Threshold) ---")
    print(classification_report(y_test, y_pred_optimal))
    
    return final_model, optimal_threshold


# --- Step 3: Main Execution Block ---

if __name__ == "__main__":
    # --- Load Data ---
    print("--- Phase 1: Loading Data ---")
    df_train_raw = pd.read_csv('train_dataset_final1.csv')
    df_val_raw = pd.read_csv('validate_dataset_final.csv')
    original_customer_ids = df_val_raw['Customer_ID']

    # --- Preprocess Training Data ---
    print("\n--- Phase 2 & 3: Preprocessing Training Data ---")
    df_train_processed, scaler, train_cols = preprocess_data(df_train_raw, is_train=True)

    # --- Split Data and Train Model ---
    X = df_train_processed[train_cols]
    y = df_train_processed['next_month_default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    final_model, optimal_threshold = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    # --- Phase 6: Process Validation Data and Generate Predictions ---
    print("\n--- Phase 6: Processing Validation Data and Generating Submission ---")
    df_val_processed = preprocess_data(df_val_raw, is_train=False, scaler=scaler, train_cols=train_cols)
    
    # Ensure validation data has the correct features for prediction
    X_val = df_val_processed[train_cols]
    
    # Generate final predictions
    final_probabilities = final_model.predict_proba(X_val)[:, 1]
    final_predictions = (final_probabilities >= optimal_threshold).astype(int)
    
    # --- Create Submission File ---
    submission_df = pd.DataFrame({
        'Customer': original_customer_ids,
        'next_month_default': final_predictions
    })
    
    submission_filename = 'submission_catboost_final.csv'
    submission_df.to_csv(submission_filename, index=False)
    
    print(f"\nSubmission file '{submission_filename}' created successfully!")
    print("\nDistribution of final predictions:")
    print(submission_df['next_month_default'].value_counts(normalize=True))
    print("\nFirst 5 predictions:")
    print(submission_df.head())