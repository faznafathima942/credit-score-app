# Save this code as 'app.py'

from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import warnings
import os # Import os for environment variables
import traceback # For detailed error logging

# Ignore warnings during prediction (optional, but cleaner logs)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- START OF CLEANING AND FEATURE ENGINEERING FUNCTION ---
# (Keep your existing clean_and_engineer_features function here - make sure it's identical to the one in train.py)
def clean_and_engineer_features(df):
    df = df.copy()
    # Drop columns
    cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Type_of_Loan']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # Clean numeric
    def clean_numeric_col(series):
        return pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    numeric_cols_to_clean = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly', 'Monthly_Balance', 'Changed_Credit_Limit', 'Delay_from_due_date'
    ]
    for col in numeric_cols_to_clean:
        if col in df.columns: df[col] = clean_numeric_col(df[col])

    # Fix Age
    if 'Age' in df.columns: df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    # Clean Categorical
    if 'Occupation' in df.columns: df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
    if 'Credit_Mix' in df.columns: df['Credit_Mix'] = df['Credit_Mix'].astype(str).str.strip('_').replace('nan', 'Unknown')
    if 'Payment_of_Min_Amount' in df.columns:
        valid_pay_values = ['Yes', 'No', 'NM']
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].astype(str)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].where(df['Payment_of_Min_Amount'].isin(valid_pay_values), np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'No_Minimum')

    # Credit History Age
    if 'Credit_History_Age' in df.columns:
        def convert_history_to_months(age_str):
            if pd.isna(age_str): return np.nan
            years, months = 0, 0
            try:
                year_match = re.search(r'(\d+)\s+Years?', str(age_str), re.IGNORECASE)
                if year_match: years = int(year_match.group(1))
                month_match = re.search(r'(\d+)\s+Months?', str(age_str), re.IGNORECASE)
                if month_match: months = int(month_match.group(1))
                if years == 0 and months == 0 and not pd.isna(age_str): return np.nan # Or handle differently
                return (years * 12) + months
            except: return np.nan
        df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_history_to_months)
        df = df.drop(columns=['Credit_History_Age'])

    # --- Feature Engineering ---
    if 'Monthly_Inhand_Salary' in df.columns:
        salary_col = pd.to_numeric(df['Monthly_Inhand_Salary'], errors='coerce').fillna(1e-6)
        salary = salary_col.replace(0, 1e-6)
        if 'Total_EMI_per_month' in df.columns:
            emi_col = pd.to_numeric(df['Total_EMI_per_month'], errors='coerce').fillna(0)
            df['Debt_to_Income_Ratio'] = emi_col / salary
        if 'Amount_invested_monthly' in df.columns:
            invest_col = pd.to_numeric(df['Amount_invested_monthly'], errors='coerce').fillna(0)
            df['Savings_Ratio'] = invest_col / salary
    if 'Annual_Income' in df.columns:
        annual_income_col = pd.to_numeric(df['Annual_Income'], errors='coerce').fillna(1e-6)
        annual_income = annual_income_col.replace(0, 1e-6)
        if 'Outstanding_Debt' in df.columns:
            debt_col = pd.to_numeric(df['Outstanding_Debt'], errors='coerce').fillna(0)
            df['Debt_to_Annual_Income'] = debt_col / annual_income
    if 'Num_of_Loan' in df.columns:
        num_loan_col = pd.to_numeric(df['Num_of_Loan'], errors='coerce').fillna(0)
        if 'Num_Credit_Card' in df.columns:
            num_card_col = pd.to_numeric(df['Num_Credit_Card'], errors='coerce').fillna(0)
            df['Total_Accounts'] = num_loan_col + num_card_col
    if 'Num_of_Delayed_Payment' in df.columns:
        delayed_pay_col = pd.to_numeric(df['Num_of_Delayed_Payment'], errors='coerce').fillna(0)
        if 'Credit_History_Months' in df.columns:
            history_months_col = pd.to_numeric(df['Credit_History_Months'], errors='coerce').fillna(1e-6)
            history_months = history_months_col.replace(0, 1e-6)
            df['Late_Payment_Rate'] = delayed_pay_col / history_months
    return df
# <--- END OF CLEANING AND FEATURE ENGINEERING FUNCTION --->


# Initialize the Flask app
app = Flask(__name__)

# --- Load Pipeline and Training Columns ---
pipeline = None
training_columns = None
pipeline_filename = 'credit_score_pipeline.joblib'
columns_filename = 'training_columns.joblib'

try:
    pipeline = joblib.load(pipeline_filename)
    print(f"Pipeline '{pipeline_filename}' loaded successfully.")
    training_columns = joblib.load(columns_filename)
    print(f"Training columns '{columns_filename}' loaded successfully. Columns: {training_columns}")
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Make sure '{pipeline_filename}' and '{columns_filename}' are in the same directory.")
except Exception as e:
    print(f"Error loading pipeline or columns: {e}")
# --- End Loading ---


# Define the prediction labels
score_labels = {0: 'Poor', 1: 'Standard', 2: 'Good'}


# --- Route for the Homepage ---
@app.route('/')
def home():
    return render_template('index.html')


# --- Route for API Predictions ---
@app.route('/predict', methods=['POST'])
def predict_api():
    # Check if pipeline and columns loaded correctly
    if pipeline is None or training_columns is None:
        print("Error: Model pipeline or training columns not loaded.")
        return jsonify({'error': 'Model configuration error. Check server logs.'}), 500

    try:
        # 1. Get Data
        data = request.get_json()
        if not data:
             print("Error: No input data received.")
             return jsonify({'error': 'No input data received'}), 400
        print(f"\n--- New Prediction Request ---")
        print(f"1. Raw input data: {data}")
        input_df_raw = pd.DataFrame([data])

        # 2. Clean and Engineer Features
        cleaned_input_df = clean_and_engineer_features(input_df_raw)
        # Convert all columns to string temporarily for logging to avoid type errors
        print(f"2. Cleaned data: {cleaned_input_df.astype(str).iloc[0].to_dict()}")

        # 3. Align and Reorder Columns
        print(f"3. Aligning columns to training order: {training_columns}")
        # Add any missing columns (that should exist after cleaning/engineering) with NaN
        for col in training_columns:
            if col not in cleaned_input_df.columns:
                print(f"   Warning: Column '{col}' missing after cleaning, adding as NaN.")
                cleaned_input_df[col] = np.nan

        # Select only the expected columns in the correct order
        try:
            input_df_aligned = cleaned_input_df[training_columns]
            # Log the first few values of the aligned data
            print(f"   Aligned data (first 5 cols): {input_df_aligned.iloc[:, :5].astype(str).iloc[0].to_dict()}")
        except KeyError as e:
             print(f"   Error: Could not align columns. Missing expected column: {e}")
             raise ValueError(f"Input data missing expected column after cleaning: {e}")


        # 4. Predict using the pipeline
        print("4. Making prediction...")
        prediction_numeric = pipeline.predict(input_df_aligned)
        print(f"   Numeric prediction: {prediction_numeric}")

        # 5. Get Probabilities (CRUCIAL FOR DEBUGGING)
        prediction_proba = None
        try:
             prediction_proba = pipeline.predict_proba(input_df_aligned)
             # Log probabilities clearly
             print(f"   Prediction probabilities (Poor, Standard, Good): {np.round(prediction_proba[0], 4)}")
        except AttributeError:
             prediction_proba = [[0.0, 0.0, 0.0]] # Placeholder if not available
             print("   predict_proba not available for this pipeline.")
        except Exception as e:
             prediction_proba = [[0.0, 0.0, 0.0]] # Placeholder on error
             print(f"   Error getting probabilities: {e}")


        # 6. Map and Format Response
        pred_num = prediction_numeric[0]
        prediction_label = score_labels.get(pred_num, "Unknown")
        print(f"5. Final Predicted Label: {prediction_label}")

        response = {
            'prediction': prediction_label,
            'prediction_numeric': int(pred_num),
            'confidence_scores': {
                'Poor': round(prediction_proba[0][0], 4),
                'Standard': round(prediction_proba[0][1], 4),
                'Good': round(prediction_proba[0][2], 4)
            }
        }
        print(f"6. Sending Response: {response}")
        return jsonify(response)

    except Exception as e:
        # Log the full error traceback for debugging
        print(f"!!! Error during prediction !!!")
        print(traceback.format_exc())
        return jsonify({'error': f'An unexpected error occurred during prediction: {str(e)}'}), 500

# Run the app
if __name__ == '__main__':
    # Render provides the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    # Host '0.0.0.0' makes it accessible externally (needed for Render)
    app.run(debug=False, host='0.0.0.0', port=port)