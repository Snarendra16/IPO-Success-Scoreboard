"""
train_local.py
Local training script for IPO Prediction (NO Streamlit, production-ready)

Run this locally or in Google Colab:
    python train_local.py

It will generate the following files:
    advanced_ensemble_model.pkl
    advanced_scaler.pkl
    advanced_selector.pkl
    selected_features.pkl
    label_encoder.pkl
    model_performance.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

DATA_PATH = "data.csv"   # Update if needed

df = pd.read_csv(DATA_PATH)
df = df.replace(['NA', '', 'NaN', 'null'], 0).fillna(0)

print(f"Loaded dataset: {df.shape}")

# ---------------------------------------------------------
# FEATURE ENGINEERING (same as production)
# ---------------------------------------------------------

def apply_feature_engineering(df):
    df = df.copy()

    core_cols = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]

    for c in core_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['PE_ROCE_Interaction'] = df['P/E'] * df['ROCE %']
    df['Profit_Margin'] = (df['Net Profit of last quarter Rs. Cr.'] /
                           (df['Quarterly Sales Rs.Cr.'] + 1)) * 100
    df['Market_Cap_to_Sales'] = df['Mar Capitalization Rs.Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)
    df['Value_Growth_Score'] = (df['Dividend Yield %'] * 0.3) + (df['ROCE %'] * 0.7)

    df['Profit_Growth_Momentum'] = df['Quarterly Profit Variation %'] * np.log1p(
        abs(df['Net Profit of last quarter Rs. Cr.']))
    df['Sales_Growth_Momentum'] = df['Quarterly Sales Variation %'] * np.log1p(df['Quarterly Sales Rs.Cr.'])
    df['Composite_Growth_Score'] = (
            df['Quarterly Profit Variation %'] + df['Quarterly Sales Variation %']) * df['ROCE %']

    df['Profit_Stability'] = 1 / (1 + abs(df['Quarterly Profit Variation %']))
    df['Size_Stability'] = np.log1p(df['Mar Capitalization Rs.Cr.'])
    df['Valuation_Risk'] = df['P/E'] / (df['ROCE %'] + 1)

    df['PE_Squared'] = df['P/E'] ** 2
    df['ROCE_Squared'] = df['ROCE %'] ** 2
    df['Profit_Var_Squared'] = df['Quarterly Profit Variation %'] ** 2
    df['Size_ROCE_Interaction'] = df['Mar Capitalization Rs.Cr.'] * df['ROCE %']

    df['Efficiency_Score'] = (df['ROCE %'] * df['Profit_Margin']) / (abs(df['P/E']) + 1)
    df['Growth_Quality'] = (df['Quarterly Profit Variation %'] * df['Profit_Margin']) / 100
    df['Market_Sentiment'] = (df['P/E'] * df['Dividend Yield %']) / 100

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    return df

# ---------------------------------------------------------
# PREPARE TARGET VARIABLE
# ---------------------------------------------------------

y = df['Classification'].astype(str).str.strip()
valid_labels = ['S', 'F', 'N']
mask = y.isin(valid_labels)

df = df[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ---------------------------------------------------------
# APPLY FEATURE ENGINEERING
# ---------------------------------------------------------

X = apply_feature_engineering(df.drop(columns=['Classification']))
print(f"Total engineered features: {X.shape[1]}")

# ---------------------------------------------------------
# SCALE FEATURES
# ---------------------------------------------------------

scaler = PowerTransformer(method='yeo-johnson')
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ---------------------------------------------------------
# ADVANCED FEATURE SELECTION (RFECV)
# ---------------------------------------------------------

print("Running RFECV for feature selection...")

selector = RFECV(
    estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
    step=1,
    cv=StratifiedKFold(3),
    scoring='f1_weighted',
    min_features_to_select=10,
    n_jobs=-1
)

X_selected = selector.fit_transform(X_scaled_df, y_encoded)
selected_features = list(X_scaled_df.columns[selector.support_])

print(f"Selected features: {len(selected_features)}")

# ---------------------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------
# BALANCE USING SMOTETOMEK
# ---------------------------------------------------------

print("Balancing classes with SMOTETomek...")

smote_tomek = SMOTETomek(
    random_state=42,
    smote=SMOTE(k_neighbors=3, random_state=42)
)

X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
print("Balanced sample size:", X_train_bal.shape)

# ---------------------------------------------------------
# BUILD ENSEMBLE MODELS
# ---------------------------------------------------------

base_models = [
    ('xgb', XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
    )),
    ('lgbm', LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )),
    ('catboost', CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        random_state=42, verbose=False
    )),
    ('rf', BalancedRandomForestClassifier(
        n_estimators=150, max_depth=8, min_samples_split=8,
        min_samples_leaf=4, random_state=42
    )),
    ('gbm', GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
]

meta_model = GradientBoostingClassifier()

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True,
    n_jobs=-1
)

print("Training Stacking Ensemble...")
stacking.fit(X_train_bal, y_train_bal)

# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

y_pred = stacking.predict(X_test)
y_test_orig = label_encoder.inverse_transform(y_test)
y_pred_orig = label_encoder.inverse_transform(y_pred)

acc = accuracy_score(y_test_orig, y_pred_orig)
f1 = f1_score(y_test_orig, y_pred_orig, average='weighted')
precision = precision_score(y_test_orig, y_pred_orig, average='weighted')
recall = recall_score(y_test_orig, y_pred_orig, average='weighted')

print("\n===== Model Performance =====")
print("Accuracy:", acc)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("\nClassification Report:")
print(classification_report(y_test_orig, y_pred_orig))

performance_dict = {
    "accuracy": acc,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "report": classification_report(y_test_orig, y_pred_orig, output_dict=True)
}

# ---------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------

pickle.dump(stacking, open("advanced_ensemble_model.pkl", "wb"))
pickle.dump(scaler, open("advanced_scaler.pkl", "wb"))
pickle.dump(selector, open("advanced_selector.pkl", "wb"))
pickle.dump(selected_features, open("selected_features.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))
pickle.dump(performance_dict, open("model_performance.pkl", "wb"))

print("\nAll artifacts saved successfully!")
print("You can now deploy your Streamlit prediction app on Render.")
