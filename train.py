import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
# *** Use ImbPipeline correctly ***
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline # Need regular Pipeline for transformers
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# *** Import SMOTENC instead of SMOTE ***
from imblearn.over_sampling import SMOTENC
import joblib
import warnings

# Ignore version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("Starting training script...")

# --- 1. LOAD DATA ---
try:
    train_df = pd.read_csv("train.csv", low_memory=False)
    comp_test_df = pd.read_csv("test.csv", low_memory=False)
    print("Successfully loaded train.csv and test.csv.")
    print(f"Original training data shape: {train_df.shape}")
    print(f"Original test data shape: {comp_test_df.shape}")
except FileNotFoundError:
    print("--- FILE NOT FOUND ERROR ---")
    print("Could not find 'train.csv' or 'test.csv' in the current directory.")
    exit()

# --- 2. CLEANING & FEATURE ENGINEERING FUNCTION ---
def clean_and_engineer_features(df):
    df = df.copy()
    cols_to_drop = ['ID', 'Customer_ID', 'Name', 'SSN', 'Month', 'Type_of_Loan']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    def clean_numeric_col(series):
        return pd.to_numeric(series.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')

    numeric_cols_to_clean = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
        'Changed_Credit_Limit', 'Delay_from_due_date'
    ]
    for col in numeric_cols_to_clean:
        if col in df.columns:
            df[col] = clean_numeric_col(df[col])
    if 'Age' in df.columns:
        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan
    if 'Occupation' in df.columns:
        df['Occupation'] = df['Occupation'].replace('_______', 'Unknown')
    if 'Credit_Mix' in df.columns:
        df['Credit_Mix'] = df['Credit_Mix'].astype(str).str.strip('_')
    if 'Payment_of_Min_Amount' in df.columns:
        valid_pay_values = ['Yes', 'No', 'NM']
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].where(df['Payment_of_Min_Amount'].isin(valid_pay_values), np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', 'No_Minimum')

    if 'Credit_History_Age' in df.columns:
        def convert_history_to_months(age_str):
            if pd.isna(age_str): return np.nan
            years, months = 0, 0
            try:
                year_match = re.search(r'(\d+)\s+Years', str(age_str))
                if year_match: years = int(year_match.group(1))
                month_match = re.search(r'(\d+)\s+Months', str(age_str))
                if month_match: months = int(month_match.group(1))
                return (years * 12) + months
            except: return np.nan
        df['Credit_History_Months'] = df['Credit_History_Age'].apply(convert_history_to_months)
        df = df.drop(columns=['Credit_History_Age'])

    # --- Feature Engineering ---
    if 'Monthly_Inhand_Salary' in df.columns:
        salary = df['Monthly_Inhand_Salary'].replace(0, 1e-6)
        if 'Total_EMI_per_month' in df.columns: df['Debt_to_Income_Ratio'] = df['Total_EMI_per_month'] / salary
        if 'Amount_invested_monthly' in df.columns: df['Savings_Ratio'] = df['Amount_invested_monthly'] / salary
    if 'Annual_Income' in df.columns:
        if 'Outstanding_Debt' in df.columns: df['Debt_to_Annual_Income'] = df['Outstanding_Debt'] / df['Annual_Income'].replace(0, 1e-6)
    if 'Num_of_Loan' in df.columns:
        if 'Num_Credit_Card' in df.columns: df['Total_Accounts'] = df['Num_of_Loan'] + df['Num_Credit_Card']
    if 'Num_of_Delayed_Payment' in df.columns:
        if 'Credit_History_Months' in df.columns: df['Late_Payment_Rate'] = df['Num_of_Delayed_Payment'] / df['Credit_History_Months'].replace(0, 1e-6)
    return df

# --- 3. APPLY CLEANING ---
print("Cleaning training data...")
cleaned_train_df = clean_and_engineer_features(train_df)
print("Cleaning competition test data...")
cleaned_comp_test_df = clean_and_engineer_features(comp_test_df)

# --- 4. PREPARE TARGET (y) and FEATURES (X) ---
score_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
cleaned_train_df['Credit_Score'] = cleaned_train_df['Credit_Score'].map(score_mapping)
cleaned_train_df = cleaned_train_df.dropna(subset=['Credit_Score'])
cleaned_train_df['Credit_Score'] = cleaned_train_df['Credit_Score'].astype(int)

X = cleaned_train_df.drop('Credit_Score', axis=1)
y = cleaned_train_df['Credit_Score']

# Align columns (important!)
X_cols = X.columns
comp_test_cols = cleaned_comp_test_df.columns
common_cols = list(set(X_cols) & set(comp_test_cols))
X = X[common_cols]
X_comp_test = cleaned_comp_test_df[common_cols]
# --- Save the final column order ---
try:
    joblib.dump(common_cols, 'training_columns.joblib')
    print(f"Saved training column order to 'training_columns.joblib'. Columns: {common_cols}")
except Exception as e:
    print(f"Error saving column order: {e}")
# --- End save column order ---

print("Data cleaning and feature engineering complete.")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 5. DEFINE PREPROCESSOR ---
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Create transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fill NaN before encoding
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

# Create the preprocessor ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough' # Keep order consistent
)

# --- 6. Create the Full ImbPipeline: Preprocessor -> SMOTENC -> Model ---

# *** Find the indices of categorical features AFTER preprocessing ***
# Need to fit the preprocessor temporarily to know the output structure
preprocessor.fit(X) # Fit on the full X to get all columns
output_features = preprocessor.get_feature_names_out()
# Categorical columns will have names like 'cat__Occupation', 'cat__Credit_Mix', etc.
categorical_feature_indices = [
    i for i, feature in enumerate(output_features)
    if any(cat_feat in feature for cat_feat in categorical_features) # Find indices based on names
]
print(f"Indices of categorical features for SMOTENC: {categorical_feature_indices}")


# Define the model to be used in the pipeline
# *** Pass the INTEGER INDICES to categorical_features ***
hgb_model = HistGradientBoostingClassifier(
    random_state=42,
    categorical_features=categorical_feature_indices # Use indices, not names
)

# Create the full pipeline using ImbPipeline
full_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),   # Preprocessing happens FIRST
    ('smotenc', SMOTENC(categorical_features=categorical_feature_indices, random_state=42, k_neighbors=5)),
    ('model', hgb_model)               # Model trains THIRD
])

print("Full ImbPipeline: Preprocessing -> SMOTENC -> HGB created.")

# --- 7. Define Hyperparameter Grid ---
print("Defining hyperparameter grid...")
param_grid = {
    'model__learning_rate': [0.05, 0.1],
    'model__max_iter': [100, 200],
    'model__max_leaf_nodes': [31, 50]
}

# --- 8. Set up Grid Search ---
grid_search = GridSearchCV(
    full_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1 # Changed n_jobs to 1
)
print("Grid Search object created.")

# --- 9. SPLIT DATA ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# --- 10. Train GridSearch ---
print("Running Grid Search with SMOTENC... (This might take several minutes)")
grid_search.fit(X_train, y_train)

print("Grid Search complete.")
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy from search: {grid_search.best_score_:.4f}")

# --- 11. EVALUATE ---
best_model_pipeline = grid_search.best_estimator_
y_pred = best_model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"\n--- Model Evaluation (on Validation Set) ---")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['Poor (0)', 'Standard (1)', 'Good (2)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print("Rows = Actual, Columns = Predicted")


# --- 12. TRAIN FINAL MODEL & SAVE ---
print("\nRe-training best model found by GridSearch on all available training data...")
best_params = grid_search.best_params_

# Create final pipeline instance
final_pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smotenc', SMOTENC(categorical_features=categorical_feature_indices, random_state=42, k_neighbors=5)),
    # *** Use INTEGER INDICES here too ***
    ('model', HistGradientBoostingClassifier(
        random_state=42,
        categorical_features=categorical_feature_indices # Use indices, not names
    ))
])

# Set best params and train on full data
final_pipeline.set_params(**best_params)
final_pipeline.fit(X, y)
print("Final model training complete.")

# Save the final pipeline
pipeline_filename = 'credit_score_pipeline.joblib'
joblib.dump(final_pipeline, pipeline_filename)

print(f"Model pipeline saved to '{pipeline_filename}'")
print("\n--- SUCCESS ---")
print(f"Model file is small and estimated accuracy is: {accuracy:.4f}")
print("You are ready to push to GitHub and deploy on Render.")