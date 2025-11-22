import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(
    page_title="Advanced IPO Success Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the IPO dataset"""
    try:
        df = pd.read_csv('data.csv')
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Replace NA values and handle missing data
        df = df.replace(['NA', '', 'NaN', 'null'], 0)
        df = df.fillna(0)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None

def advanced_feature_engineering(df):
    """Advanced feature engineering with domain knowledge"""
    
    # Define core feature columns
    feature_columns = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]
    
    # Extract features and target
    X = df[feature_columns].copy()
    y = df['Classification'].copy()
    
    # Convert features to numeric
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Convert target and filter valid classes
    y = y.astype(str).str.strip()
    valid_classes = ['S', 'F', 'N']
    mask = y.isin(valid_classes)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Create mapping for reference
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    st.info(f"📊 After preprocessing: {len(X)} samples, {len(feature_columns)} features")
    st.write(f"**Class Distribution:** {dict(y.value_counts())}")
    st.write(f"**Label Mapping:** {label_mapping}")
    
    # ============================================================================
    # ADVANCED FEATURE ENGINEERING
    # ============================================================================
    
    # Financial Ratios and Metrics
    X['PE_ROCE_Interaction'] = X['P/E'] * X['ROCE %']
    X['Profit_Margin'] = (X['Net Profit of last quarter Rs. Cr.'] / (X['Quarterly Sales Rs.Cr.'] + 1)) * 100
    X['Market_Cap_to_Sales'] = X['Mar Capitalization Rs.Cr.'] / (X['Quarterly Sales Rs.Cr.'] + 1)
    X['Value_Growth_Score'] = (X['Dividend Yield %'] * 0.3) + (X['ROCE %'] * 0.7)
    
    # Growth and Momentum Indicators
    X['Profit_Growth_Momentum'] = X['Quarterly Profit Variation %'] * np.log1p(abs(X['Net Profit of last quarter Rs. Cr.']))
    X['Sales_Growth_Momentum'] = X['Quarterly Sales Variation %'] * np.log1p(X['Quarterly Sales Rs.Cr.'])
    X['Composite_Growth_Score'] = (X['Quarterly Profit Variation %'] + X['Quarterly Sales Variation %']) * X['ROCE %']
    
    # Stability and Risk Measures
    X['Profit_Stability'] = 1 / (1 + abs(X['Quarterly Profit Variation %']))
    X['Size_Stability'] = np.log1p(X['Mar Capitalization Rs.Cr.'])
    X['Valuation_Risk'] = X['P/E'] / (X['ROCE %'] + 1)
    
    # Polynomial and Interaction Features
    X['PE_Squared'] = X['P/E'] ** 2
    X['ROCE_Squared'] = X['ROCE %'] ** 2
    X['Profit_Var_Squared'] = X['Quarterly Profit Variation %'] ** 2
    X['Size_ROCE_Interaction'] = X['Mar Capitalization Rs.Cr.'] * X['ROCE %']
    
    # Advanced Financial Engineering
    X['Efficiency_Score'] = (X['ROCE %'] * X['Profit_Margin']) / (abs(X['P/E']) + 1)
    X['Growth_Quality'] = (X['Quarterly Profit Variation %'] * X['Profit_Margin']) / 100
    X['Market_Sentiment'] = (X['P/E'] * X['Dividend Yield %']) / 100
    
    # Handle any new NaN or infinite values
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)
    
    st.success(f"🔧 Advanced feature engineering completed: {X.shape[1]} total features")
    
    return X, y_encoded, label_encoder, feature_columns

def create_advanced_ensemble(label_encoder):
    """Create an advanced ensemble of diverse models with proper label handling"""
    
    # Get the number of classes from the label encoder
    n_classes = len(label_encoder.classes_)
    
    # Base models
    base_models = [
        ('xgb', XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )),
        ('catboost', CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )),
        ('rf', BalancedRandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        )),
        ('gbm', GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]
    
    # Meta-learner
    meta_learner = LogisticRegression(
        C=0.1,
        random_state=42,
        max_iter=1000,
        multi_class='multinomial'
    )
    
    # Create stacking ensemble
    stacking_ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        passthrough=True,
        n_jobs=-1
    )
    
    # Also create voting ensemble as backup
    voting_ensemble = VotingClassifier(
        estimators=base_models[:3],  # Use top 3 models for voting
        voting='soft',
        weights=[1.2, 1.0, 1.1]
    )
    
    return stacking_ensemble, voting_ensemble, base_models

def evaluate_individual_models(base_models, X_train, y_train, X_test, y_test, label_encoder):
    """Evaluate individual base models and return their performance"""
    model_performance = {}
    
    for name, model in base_models:
        with st.spinner(f"Training {name}..."):
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Convert predictions back to original labels for evaluation
            y_test_original = label_encoder.inverse_transform(y_test)
            y_pred_original = label_encoder.inverse_transform(y_pred)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_original, y_pred_original)
            f1 = f1_score(y_test_original, y_pred_original, average='weighted')
            precision = precision_score(y_test_original, y_pred_original, average='weighted', zero_division=0)
            recall = recall_score(y_test_original, y_pred_original, average='weighted', zero_division=0)
            
            # Store performance
            model_performance[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'predictions': y_pred_original,
                'probabilities': y_pred_proba
            }
    
    return model_performance

def train_advanced_model(X, y_encoded, label_encoder):
    """Train the advanced ensemble model with optimization"""
    
    with st.spinner("🔄 Performing advanced feature selection..."):
        # Initial scaling
        scaler = PowerTransformer(method='yeo-johnson')
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Advanced feature selection using RFECV with Gradient Boosting
        selector = RFECV(
            estimator=GradientBoostingClassifier(n_estimators=50, random_state=42),
            step=1,
            cv=StratifiedKFold(3),
            scoring='f1_weighted',
            min_features_to_select=10,
            n_jobs=-1
        )
        X_selected = selector.fit_transform(X_scaled_df, y_encoded)
        selected_features = X_scaled_df.columns[selector.support_].tolist()
        
        st.success(f"🎯 Feature selection completed: {len(selected_features)} features selected")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Advanced balancing
    smote_tomek = SMOTETomek(
        random_state=42,
        smote=SMOTE(k_neighbors=3, random_state=42)
    )
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)
    
    st.info(f"⚖️ Data balanced: {len(X_train_balanced)} training samples")
    
    # Create ensembles
    stacking_ensemble, voting_ensemble, base_models = create_advanced_ensemble(label_encoder)
    
    # Evaluate individual models
    st.info("📊 Evaluating individual model performance...")
    individual_performance = evaluate_individual_models(
        [(name, model) for name, model in base_models], 
        X_train_balanced, y_train_balanced, X_test, y_test, label_encoder
    )
    
    # Train stacking ensemble
    with st.spinner("🏋️ Training advanced stacking ensemble..."):
        stacking_ensemble.fit(X_train_balanced, y_train_balanced)
        stacking_pred = stacking_ensemble.predict(X_test)
        stacking_pred_original = label_encoder.inverse_transform(stacking_pred)
        y_test_original = label_encoder.inverse_transform(y_test)
        stacking_accuracy = accuracy_score(y_test_original, stacking_pred_original)
        stacking_f1 = f1_score(y_test_original, stacking_pred_original, average='weighted')
        stacking_precision = precision_score(y_test_original, stacking_pred_original, average='weighted', zero_division=0)
        stacking_recall = recall_score(y_test_original, stacking_pred_original, average='weighted', zero_division=0)
    
    # Train voting ensemble
    with st.spinner("🏋️ Training voting ensemble..."):
        voting_ensemble.fit(X_train_balanced, y_train_balanced)
        voting_pred = voting_ensemble.predict(X_test)
        voting_pred_original = label_encoder.inverse_transform(voting_pred)
        voting_accuracy = accuracy_score(y_test_original, voting_pred_original)
        voting_f1 = f1_score(y_test_original, voting_pred_original, average='weighted')
        voting_precision = precision_score(y_test_original, voting_pred_original, average='weighted', zero_division=0)
        voting_recall = recall_score(y_test_original, voting_pred_original, average='weighted', zero_division=0)
    
    # Store ensemble performances
    individual_performance['stacking'] = {
        'model': stacking_ensemble,
        'accuracy': stacking_accuracy,
        'f1_score': stacking_f1,
        'precision': stacking_precision,
        'recall': stacking_recall,
        'predictions': stacking_pred_original
    }
    
    individual_performance['voting'] = {
        'model': voting_ensemble,
        'accuracy': voting_accuracy,
        'f1_score': voting_f1,
        'precision': voting_precision,
        'recall': voting_recall,
        'predictions': voting_pred_original
    }
    
    # Choose the best ensemble
    if stacking_accuracy >= voting_accuracy:
        best_ensemble = stacking_ensemble
        ensemble_type = "Stacking Ensemble"
        ensemble_score = stacking_accuracy
    else:
        best_ensemble = voting_ensemble
        ensemble_type = "Voting Ensemble"
        ensemble_score = voting_accuracy
    
    # Enhanced styling for the success message
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 20px 0;
        ">
            <h2 style="margin: 0; font-size: 24px;">🎉 Model Training Completed!</h2>
            <p style="margin: 10px 0 0 0; font-size: 18px;">
                Best Model: <strong>{ensemble_type}</strong> | Accuracy: <strong>{ensemble_score:.3f}</strong>
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    return best_ensemble, scaler, selector, X_train, X_test, y_train, y_test, selected_features, individual_performance, label_encoder

def predict_with_gmp_advanced(model, scaler, selector, input_features, gmp_ratio, selected_features, label_encoder):
    """
    Advanced prediction with GMP integration and confidence calibration
    """
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Apply the same feature engineering as training
        input_df = apply_feature_engineering(input_df)
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
        
        # Feature selection
        input_selected = selector.transform(input_scaled_df)
        input_final = pd.DataFrame(input_selected, columns=selected_features)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_final)[0]
        
        # Get predicted class (numerical) and convert to original label
        predicted_class_encoded = model.predict(input_final)[0]
        predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
        
        # Get base success probability
        success_idx = list(label_encoder.classes_).index('S')
        base_success_prob = probabilities[success_idx] * 100
        
        # Advanced GMP integration with dynamic weighting
        if gmp_ratio > 0:
            # Dynamic GMP weight based on market conditions
            gmp_weight = calculate_dynamic_gmp_weight(gmp_ratio)
            gmp_contribution = gmp_weight * gmp_ratio
            
            # Apply GMP adjustment with diminishing returns
            adjusted_score = base_success_prob + gmp_contribution
            
            # Cap at 95 (leaving room for uncertainty)
            adjusted_score = min(adjusted_score, 95)
        else:
            adjusted_score = base_success_prob
            gmp_contribution = 0
        
        # Calculate confidence based on probability distribution
        confidence = calculate_prediction_confidence(probabilities, label_encoder.classes_)
        
        # Determine final classification with calibrated thresholds
        final_class = determine_final_classification(adjusted_score, confidence)
        
        # Create probabilities dictionary with original labels
        prob_dict = dict(zip(label_encoder.classes_, probabilities))
        
        return {
            'predicted_class': final_class,
            'base_score': base_success_prob,
            'adjusted_score': adjusted_score,
            'gmp_contribution': gmp_contribution,
            'confidence': confidence * 100,
            'probabilities': prob_dict
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def calculate_dynamic_gmp_weight(gmp_ratio):
    """Calculate dynamic weight for GMP based on ratio value"""
    if gmp_ratio <= 20:
        return 0.1  # Low weight for low GMP
    elif gmp_ratio <= 50:
        return 0.2  # Medium weight
    elif gmp_ratio <= 100:
        return 0.3  # High weight
    else:
        return 0.25  # Slightly reduced weight for very high GMP (might be speculative)

def calculate_prediction_confidence(probabilities, classes):
    """Calculate prediction confidence based on probability distribution"""
    sorted_probs = sorted(probabilities, reverse=True)
    if len(sorted_probs) >= 2:
        # Confidence is high if top probability is significantly higher than second
        confidence = sorted_probs[0] - sorted_probs[1]
    else:
        confidence = sorted_probs[0]
    
    return min(confidence * 1.5, 1.0)  # Scale confidence

def determine_final_classification(adjusted_score, confidence):
    """Determine final classification with calibrated thresholds"""
    
    # Adjust thresholds based on confidence
    confidence_adjustment = (1 - confidence) * 10
    
    if adjusted_score >= (75 - confidence_adjustment):
        return 'S'  # Success
    elif adjusted_score >= (40 - confidence_adjustment):
        return 'N'  # Normal
    else:
        return 'F'  # Fail

def apply_feature_engineering(df):
    """Apply the same feature engineering as during training"""
    
    # Financial Ratios and Metrics
    df['PE_ROCE_Interaction'] = df['P/E'] * df['ROCE %']
    df['Profit_Margin'] = (df['Net Profit of last quarter Rs. Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)) * 100
    df['Market_Cap_to_Sales'] = df['Mar Capitalization Rs.Cr.'] / (df['Quarterly Sales Rs.Cr.'] + 1)
    df['Value_Growth_Score'] = (df['Dividend Yield %'] * 0.3) + (df['ROCE %'] * 0.7)
    
    # Growth and Momentum Indicators
    df['Profit_Growth_Momentum'] = df['Quarterly Profit Variation %'] * np.log1p(abs(df['Net Profit of last quarter Rs. Cr.']))
    df['Sales_Growth_Momentum'] = df['Quarterly Sales Variation %'] * np.log1p(df['Quarterly Sales Rs.Cr.'])
    df['Composite_Growth_Score'] = (df['Quarterly Profit Variation %'] + df['Quarterly Sales Variation %']) * df['ROCE %']
    
    # Stability and Risk Measures
    df['Profit_Stability'] = 1 / (1 + abs(df['Quarterly Profit Variation %']))
    df['Size_Stability'] = np.log1p(df['Mar Capitalization Rs.Cr.'])
    df['Valuation_Risk'] = df['P/E'] / (df['ROCE %'] + 1)
    
    # Polynomial and Interaction Features
    df['PE_Squared'] = df['P/E'] ** 2
    df['ROCE_Squared'] = df['ROCE %'] ** 2
    df['Profit_Var_Squared'] = df['Quarterly Profit Variation %'] ** 2
    df['Size_ROCE_Interaction'] = df['Mar Capitalization Rs.Cr.'] * df['ROCE %']
    
    # Advanced Financial Engineering
    df['Efficiency_Score'] = (df['ROCE %'] * df['Profit_Margin']) / (abs(df['P/E']) + 1)
    df['Growth_Quality'] = (df['Quarterly Profit Variation %'] * df['Profit_Margin']) / 100
    df['Market_Sentiment'] = (df['P/E'] * df['Dividend Yield %']) / 100
    
    # Handle any new NaN or infinite values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

def display_correlation_heatmaps(df, X_engineered):
    """Display correlation heatmaps for original and engineered features"""
    
    st.header("🔥 Correlation Analysis")
    
    # Original features correlation
    st.subheader("📊 Original Features Correlation Heatmap")
    
    # Select only original numeric features from the dataframe (exclude Classification)
    numeric_features = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %'
    ]
    
    # Filter numeric features that exist in the dataframe
    existing_numeric_features = [col for col in numeric_features if col in df.columns]
    original_df_numeric = df[existing_numeric_features].copy()
    
    # Ensure all columns are numeric
    for col in original_df_numeric.columns:
        original_df_numeric[col] = pd.to_numeric(original_df_numeric[col], errors='coerce')
    
    # Fill NaN values
    original_df_numeric = original_df_numeric.fillna(0)
    
    # Calculate correlation matrix for original features
    if not original_df_numeric.empty:
        original_corr = original_df_numeric.corr()
        
        # Plot original features correlation heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(original_corr, dtype=bool))
        sns.heatmap(original_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix - Original Features", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for original features correlation heatmap.")
    
    # Engineered features correlation
    st.subheader("🔧 Engineered Features Correlation Heatmap")
    
    # Ensure engineered features are all numeric
    X_engineered_numeric = X_engineered.copy()
    for col in X_engineered_numeric.columns:
        X_engineered_numeric[col] = pd.to_numeric(X_engineered_numeric[col], errors='coerce')
    
    # Fill NaN values
    X_engineered_numeric = X_engineered_numeric.fillna(0)
    
    # Calculate correlation matrix for engineered features
    if not X_engineered_numeric.empty:
        engineered_corr = X_engineered_numeric.corr()
        
        # Plot engineered features correlation heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(engineered_corr, dtype=bool))
        sns.heatmap(engineered_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix - Engineered Features", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for engineered features correlation heatmap.")
    
    # Correlation insights
    st.subheader("📈 Correlation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Features:**")
        st.write("- Shows relationships between basic financial metrics")
        st.write("- Helps identify multicollinearity in raw data")
        st.write("- Useful for understanding fundamental relationships")
    
    with col2:
        st.write("**Engineered Features:**")
        st.write("- Reveals complex interactions between derived features")
        st.write("- Highlights domain-specific financial relationships")
        st.write("- Guides feature selection for model training")
    """Display correlation heatmaps for original and engineered features"""
    
    st.header("🔥 Correlation Analysis")
    
    # Original features correlation
    st.subheader("📊 Original Features Correlation Heatmap")
    
    # Select only original features from the dataframe
    original_features = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %', 'Classification'
    ]
    
    # Filter original features that exist in the dataframe
    existing_original_features = [col for col in original_features if col in df.columns]
    original_df = df[existing_original_features].copy()
    
    # Handle Classification column - ensure it's properly formatted
    if 'Classification' in original_df.columns:
        # Convert to string and strip whitespace
        original_df['Classification'] = original_df['Classification'].apply(
            lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
        )
        
        # Create a more robust encoding without LabelEncoder
        unique_classes = original_df['Classification'].unique()
        class_mapping = {cls: i for i, cls in enumerate(unique_classes)}
        original_df['Classification_encoded'] = original_df['Classification'].map(class_mapping)
        
        original_df_numeric = original_df.drop('Classification', axis=1)
    else:
        original_df_numeric = original_df
    
    # Ensure all columns are numeric and handle any non-numeric values
    for col in original_df_numeric.columns:
        original_df_numeric[col] = pd.to_numeric(original_df_numeric[col], errors='coerce')
    
    # Drop any remaining non-numeric columns and handle NaN values
    original_df_numeric = original_df_numeric.select_dtypes(include=[np.number])
    original_df_numeric = original_df_numeric.fillna(0)
    
    # Calculate correlation matrix for original features
    original_corr = original_df_numeric.corr()
    
    # Plot original features correlation heatmap
    if not original_corr.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(original_corr, dtype=bool))
        sns.heatmap(original_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix - Original Features", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for original features correlation heatmap.")
    
    # Engineered features correlation
    st.subheader("🔧 Engineered Features Correlation Heatmap")
    
    # Ensure engineered features are all numeric
    X_engineered_numeric = X_engineered.copy()
    for col in X_engineered_numeric.columns:
        X_engineered_numeric[col] = pd.to_numeric(X_engineered_numeric[col], errors='coerce')
    
    # Drop any remaining non-numeric columns and handle NaN values
    X_engineered_numeric = X_engineered_numeric.select_dtypes(include=[np.number])
    X_engineered_numeric = X_engineered_numeric.fillna(0)
    
    # Calculate correlation matrix for engineered features
    engineered_corr = X_engineered_numeric.corr()
    
    # Plot engineered features correlation heatmap
    if not engineered_corr.empty:
        fig, ax = plt.subplots(figsize=(16, 14))
        mask = np.triu(np.ones_like(engineered_corr, dtype=bool))
        sns.heatmap(engineered_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix - Engineered Features", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
    else:
        st.warning("No numeric data available for engineered features correlation heatmap.")
    
    # Correlation insights
    st.subheader("📈 Correlation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Features:**")
        st.write("- Shows relationships between basic financial metrics")
        st.write("- Helps identify multicollinearity in raw data")
        st.write("- Useful for understanding fundamental relationships")
    
    with col2:
        st.write("**Engineered Features:**")
        st.write("- Reveals complex interactions between derived features")
        st.write("- Highlights domain-specific financial relationships")
        st.write("- Guides feature selection for model training")
    """Display correlation heatmaps for original and engineered features"""
    
    st.header("🔥 Correlation Analysis")
    
    # Original features correlation
    st.subheader("📊 Original Features Correlation Heatmap")
    
    # Select only original features from the dataframe
    original_features = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %', 'Classification'
    ]
    
    # Filter original features that exist in the dataframe
    existing_original_features = [col for col in original_features if col in df.columns]
    original_df = df[existing_original_features].copy()
    
    # Convert Classification to string and then to numeric for correlation
    if 'Classification' in original_df.columns:
        # Convert to string first to handle mixed types
        original_df['Classification'] = original_df['Classification'].astype(str)
        le = LabelEncoder()
        original_df['Classification_encoded'] = le.fit_transform(original_df['Classification'])
        original_df_numeric = original_df.drop('Classification', axis=1)
    else:
        original_df_numeric = original_df
    
    # Ensure all columns are numeric
    for col in original_df_numeric.columns:
        original_df_numeric[col] = pd.to_numeric(original_df_numeric[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    original_df_numeric = original_df_numeric.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix for original features
    original_corr = original_df_numeric.corr()
    
    # Plot original features correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(original_corr, dtype=bool))
    sns.heatmap(original_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Original Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    # Engineered features correlation
    st.subheader("🔧 Engineered Features Correlation Heatmap")
    
    # Ensure engineered features are all numeric
    X_engineered_numeric = X_engineered.copy()
    for col in X_engineered_numeric.columns:
        X_engineered_numeric[col] = pd.to_numeric(X_engineered_numeric[col], errors='coerce')
    
    # Drop any remaining non-numeric columns
    X_engineered_numeric = X_engineered_numeric.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix for engineered features
    engineered_corr = X_engineered_numeric.corr()
    
    # Plot engineered features correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(engineered_corr, dtype=bool))
    sns.heatmap(engineered_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Engineered Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    # Correlation insights
    st.subheader("📈 Correlation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Features:**")
        st.write("- Shows relationships between basic financial metrics")
        st.write("- Helps identify multicollinearity in raw data")
        st.write("- Useful for understanding fundamental relationships")
    
    with col2:
        st.write("**Engineered Features:**")
        st.write("- Reveals complex interactions between derived features")
        st.write("- Highlights domain-specific financial relationships")
        st.write("- Guides feature selection for model training")
    """Display correlation heatmaps for original and engineered features"""
    
    st.header("🔥 Correlation Analysis")
    
    # Original features correlation
    st.subheader("📊 Original Features Correlation Heatmap")
    
    # Select only original features from the dataframe
    original_features = [
        'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
        'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
        'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
        'Issue Price (Rs)', 'ROCE %', 'Classification'
    ]
    
    # Filter original features that exist in the dataframe
    existing_original_features = [col for col in original_features if col in df.columns]
    original_df = df[existing_original_features].copy()
    
    # Convert Classification to numeric for correlation
    if 'Classification' in original_df.columns:
        le = LabelEncoder()
        original_df['Classification_encoded'] = le.fit_transform(original_df['Classification'])
        original_df_numeric = original_df.drop('Classification', axis=1)
    else:
        original_df_numeric = original_df
    
    # Calculate correlation matrix for original features
    original_corr = original_df_numeric.corr()
    
    # Plot original features correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(original_corr, dtype=bool))
    sns.heatmap(original_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Original Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    # Engineered features correlation
    st.subheader("🔧 Engineered Features Correlation Heatmap")
    
    # Calculate correlation matrix for engineered features
    engineered_corr = X_engineered.corr()
    
    # Plot engineered features correlation heatmap
    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(engineered_corr, dtype=bool))
    sns.heatmap(engineered_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Engineered Features", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    # Correlation insights
    st.subheader("📈 Correlation Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Features:**")
        st.write("- Shows relationships between basic financial metrics")
        st.write("- Helps identify multicollinearity in raw data")
        st.write("- Useful for understanding fundamental relationships")
    
    with col2:
        st.write("**Engineered Features:**")
        st.write("- Reveals complex interactions between derived features")
        st.write("- Highlights domain-specific financial relationships")
        st.write("- Guides feature selection for model training")

def display_model_performance(individual_performance, y_test_original):
    """Display comprehensive model performance summary"""
    
    st.header("📊 Comprehensive Model Performance Analysis")
    
    # Create performance dataframe
    performance_data = []
    for model_name, metrics in individual_performance.items():
        if model_name not in ['stacking', 'voting']:  # Base models
            performance_data.append({
                'Model': model_name.upper(),
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}"
            })
        else:  # Ensemble models
            performance_data.append({
                'Model': f"{model_name.title()} Ensemble",
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'F1-Score': f"{metrics['f1_score']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}"
            })
    
    # Display performance table
    st.subheader("🎯 Model Accuracy Comparison")
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # Visual comparison
    st.subheader("📈 Model Performance Visualization")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    base_models = [k for k in individual_performance.keys() if k not in ['stacking', 'voting']]
    base_accuracies = [individual_performance[model]['accuracy'] for model in base_models]
    
    axes[0, 0].bar(base_models, base_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    axes[0, 0].set_title('Base Models Accuracy')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Ensemble comparison
    ensemble_models = ['stacking', 'voting']
    ensemble_accuracies = [individual_performance[model]['accuracy'] for model in ensemble_models]
    
    axes[0, 1].bar(ensemble_models, ensemble_accuracies, color=['#A8E6CF', '#DCEDC1'])
    axes[0, 1].set_title('Ensemble Models Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    
    # F1-Score comparison
    base_f1 = [individual_performance[model]['f1_score'] for model in base_models]
    axes[1, 0].bar(base_models, base_f1, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    axes[1, 0].set_title('Base Models F1-Score')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Best model highlight
    best_model = max(individual_performance.items(), key=lambda x: x[1]['accuracy'])
    axes[1, 1].text(0.1, 0.5, f"🏆 Best Model: {best_model[0].upper()}\nAccuracy: {best_model[1]['accuracy']:.3f}", 
                   fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Detailed metrics for best model
    st.subheader("🥇 Best Performing Model Details")
    best_model_name = best_model[0]
    best_metrics = best_model[1]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.3f}")
    with col2:
        st.metric("F1-Score", f"{best_metrics['f1_score']:.3f}")
    with col3:
        st.metric("Precision", f"{best_metrics['precision']:.3f}")
    with col4:
        st.metric("Recall", f"{best_metrics['recall']:.3f}")
    
    # Confusion Matrix for best model
    st.subheader("📋 Confusion Matrix - Best Model")
    if 'predictions' in best_metrics:
        cm = confusion_matrix(y_test_original, best_metrics['predictions'], labels=['S', 'F', 'N'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Success', 'Fail', 'Normal'],
                   yticklabels=['Success', 'Fail', 'Normal'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {best_model_name.upper()}')
        st.pyplot(fig)

def main():
    st.title("🎯 Advanced IPO Success Prediction Platform")
    st.markdown("---")
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = None
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Model Training", "IPO Prediction", "Model Analysis", "Data Analysis"]
    )
    
    if app_mode == "Model Training":
        show_model_training()
    elif app_mode == "IPO Prediction":
        show_ipo_prediction()
    elif app_mode == "Data Analysis":
        show_data_analysis()
    else:
        show_model_analysis()

def show_model_training():
    st.header("🔬 Advanced Model Training")
    
    if st.button("🚀 Train Advanced Ensemble Model"):
        with st.spinner("Loading and preprocessing data..."):
            df = load_and_preprocess_data()
            
        if df is not None:
            with st.spinner("Performing advanced feature engineering..."):
                X, y_encoded, label_encoder, original_features = advanced_feature_engineering(df)
            
            with st.spinner("Training advanced ensemble model..."):
                model, scaler, selector, X_train, X_test, y_train, y_test, selected_features, individual_performance, label_encoder = train_advanced_model(X, y_encoded, label_encoder)
                
                # Save models and encoders
                with open('advanced_ensemble_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
                with open('advanced_scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                with open('advanced_selector.pkl', 'wb') as f:
                    pickle.dump(selector, f)
                with open('selected_features.pkl', 'wb') as f:
                    pickle.dump(selected_features, f)
                with open('model_performance.pkl', 'wb') as f:
                    pickle.dump(individual_performance, f)
                with open('label_encoder.pkl', 'wb') as f:
                    pickle.dump(label_encoder, f)
                
                st.session_state.model_trained = True
                st.session_state.model_performance = individual_performance
                st.success("✅ Advanced model trained and saved successfully!")

def show_ipo_prediction():
    st.header("🔮 IPO Success Prediction")
    
    # Load trained models and encoders
    try:
        with open('advanced_ensemble_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('advanced_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('advanced_selector.pkl', 'rb') as f:
            selector = pickle.load(f)
        with open('selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Models not found. Please train the model first.")
        return
    
    st.info("""
    **🎯 Classification System:**
    - **Success (S)**: Strong fundamentals + positive market sentiment
    - **Normal (N)**: Moderate potential with acceptable risk
    - **Fail (F)**: High risk or weak fundamentals
    
    **📊 GMP Integration:** Dynamically weighted based on market conditions
    """)
    
    # GMP Input First
    st.subheader("📈 Step 1: Enter Grey Market Premium (GMP)")
    gmp_ratio = st.number_input(
        "GMP Ratio",
        min_value=0.0,
        max_value=200.0,
        value=0.0,
        step=5.0,
        help="Enter the Grey Market Premium ratio (0 if not available)"
    )
    
    if gmp_ratio > 0:
        st.success(f"✅ GMP Ratio: {gmp_ratio} (Will contribute to prediction)")
    
    # IPO Details Form
    with st.form("advanced_ipo_prediction"):
        st.subheader("📋 Step 2: Enter IPO Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pe_ratio = st.number_input("P/E Ratio", min_value=0.0, value=20.0, step=1.0)
            market_cap = st.number_input("Market Capitalization (Rs. Cr.)", min_value=0.0, value=1500.0, step=100.0)
            dividend_yield = st.number_input("Dividend Yield %", min_value=0.0, value=2.5, step=0.1)
        
        with col2:
            net_profit = st.number_input("Net Profit last quarter (Rs. Cr.)", min_value=0.0, value=75.0, step=10.0)
            profit_variation = st.number_input("Quarterly Profit Variation %", value=15.0, step=1.0)
            quarterly_sales = st.number_input("Quarterly Sales (Rs. Cr.)", min_value=0.0, value=600.0, step=50.0)
        
        with col3:
            sales_variation = st.number_input("Quarterly Sales Variation %", value=12.0, step=1.0)
            issue_price = st.number_input("Issue Price (Rs)", min_value=0.0, value=120.0, step=10.0)
            roce = st.number_input("ROCE %", value=18.0, step=1.0)
        
        submitted = st.form_submit_button("🎯 Predict IPO Success")
    
    if submitted:
        input_features = {
            'P/E': pe_ratio,
            'Mar Capitalization Rs.Cr.': market_cap,
            'Dividend Yield %': dividend_yield,
            'Net Profit of last quarter Rs. Cr.': net_profit,
            'Quarterly Profit Variation %': profit_variation,
            'Quarterly Sales Rs.Cr.': quarterly_sales,
            'Quarterly Sales Variation %': sales_variation,
            'Issue Price (Rs)': issue_price,
            'ROCE %': roce
        }
        
        with st.spinner("🔄 Performing advanced analysis..."):
            result = predict_with_gmp_advanced(
                model, scaler, selector, input_features, gmp_ratio, selected_features, label_encoder
            )
        
        if result:
            display_advanced_results(result, gmp_ratio)

def display_advanced_results(result, gmp_ratio):
    """Display advanced prediction results"""
    
    st.subheader("🎯 Prediction Results")
    
    # Color coding based on prediction
    if result['predicted_class'] == 'S':
        color = "green"
        icon = "✅"
        message = "HIGH SUCCESS POTENTIAL"
    elif result['predicted_class'] == 'N':
        color = "blue"
        icon = "ℹ️"
        message = "MODERATE POTENTIAL"
    else:
        color = "red"
        icon = "❌"
        message = "HIGH RISK"
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Classification", f"{icon} {result['predicted_class']}")
    
    with col2:
        st.metric("Adjusted Score", f"{result['adjusted_score']:.1f}")
    
    with col3:
        st.metric("Confidence", f"{result['confidence']:.1f}%")
    
    with col4:
        st.metric("Base Score", f"{result['base_score']:.1f}")
    
    # Detailed breakdown
    st.subheader("📊 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Score Breakdown:**")
        st.write(f"- Base Model Score: {result['base_score']:.1f}")
        if gmp_ratio > 0:
            st.write(f"- GMP Contribution: +{result['gmp_contribution']:.1f}")
        st.write(f"- **Final Score: {result['adjusted_score']:.1f}**")
        
        st.write("**Probability Distribution:**")
        for cls, prob in result['probabilities'].items():
            st.write(f"- {cls}: {prob*100:.1f}%")
    
    with col2:
        st.write("**Classification Thresholds:**")
        st.write("- Success (S): ≥ 70")
        st.write("- Normal (N): 40 - 69")
        st.write("- Fail (F): < 40")
        st.write(f"- **Your Score: {result['adjusted_score']:.1f}**")
    
    # Visual scale
    st.subheader("📈 Performance Scale")
    create_visual_scale(result['adjusted_score'])
    
    # Recommendation
    st.subheader("💡 Investment Recommendation")
    display_recommendation(result, gmp_ratio)

def create_visual_scale(score):
    """Create a visual performance scale"""
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create gradient background
    x = np.linspace(0, 100, 100)
    colors = ['red'] * 40 + ['blue'] * 30 + ['green'] * 30
    
    for i in range(len(x)-1):
        ax.axvspan(x[i], x[i+1], color=colors[i], alpha=0.3)
    
    # Add score marker
    ax.axvline(x=score, color='black', linewidth=3, label=f'Score: {score:.1f}')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Performance Score')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add threshold labels
    ax.text(20, 0.5, 'FAIL', ha='center', va='center', fontweight='bold')
    ax.text(55, 0.5, 'NORMAL', ha='center', va='center', fontweight='bold')
    ax.text(85, 0.5, 'SUCCESS', ha='center', va='center', fontweight='bold')
    
    st.pyplot(fig)

def display_recommendation(result, gmp_ratio):
    """Display investment recommendation"""
    
    if result['predicted_class'] == 'S':
        st.success("""
        **🎉 STRONG BUY RECOMMENDATION**
        
        **Key Strengths:**
        - Strong fundamental metrics
        - Positive growth indicators
        - Favorable risk-reward ratio
        
        **Action:** Consider applying with high allocation. Monitor market conditions.
        """)
        
    elif result['predicted_class'] == 'N':
        st.warning("""
        **⚖️ CAUTIOUS APPROACH RECOMMENDATION**
        
        **Considerations:**
        - Moderate fundamentals
        - Acceptable risk levels
        - Requires careful analysis
        
        **Action:** Consider applying with moderate allocation. Review company details.
        """)
        
    else:
        st.error("""
        **🚨 HIGH RISK - AVOID RECOMMENDATION**
        
        **Concerns Identified:**
        - Weak fundamental metrics
        - High risk factors
        - Poor growth indicators
        
        **Action:** Avoid application. Wait for better opportunities.
        """)
    
    # GMP-specific advice
    if gmp_ratio > 50:
        st.info(f"💡 **GMP Insight:** High GMP ({gmp_ratio}) indicates strong market demand. This has been factored into the positive adjustment.")

def show_data_analysis():
    """Display data analysis including correlation heatmaps"""
    st.header("📈 Data Analysis & Feature Correlation")
    
    df = load_and_preprocess_data()
    if df is not None:
        with st.spinner("Performing feature engineering for correlation analysis..."):
            X_engineered, _, _, _ = advanced_feature_engineering(df)
        
        # Display correlation heatmaps
        display_correlation_heatmaps(df, X_engineered)
        
        # Additional data insights
        st.header("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Original Features", 9)  # Fixed number of original features
        with col3:
            st.metric("Engineered Features", X_engineered.shape[1])
        
        # Class distribution
        st.subheader("🎯 Target Variable Distribution")
        
        # Clean the Classification column
        classification_clean = df['Classification'].copy()
        classification_clean = classification_clean.apply(
            lambda x: str(x).strip() if pd.notna(x) else 'Unknown'
        )
        
        class_dist = classification_clean.value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_dist.plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'])
        plt.title('IPO Classification Distribution')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        # Show class distribution as table
        st.write("**Class Distribution Table:**")
        st.dataframe(class_dist)
    """Display data analysis including correlation heatmaps"""
    st.header("📈 Data Analysis & Feature Correlation")
    
    df = load_and_preprocess_data()
    if df is not None:
        with st.spinner("Performing feature engineering for correlation analysis..."):
            X_engineered, _, _, _ = advanced_feature_engineering(df)
        
        # Display correlation heatmaps
        display_correlation_heatmaps(df, X_engineered)
        
        # Additional data insights
        st.header("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Original Features", len([
                'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
                'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
                'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
                'Issue Price (Rs)', 'ROCE %'
            ]))
        with col3:
            st.metric("Engineered Features", X_engineered.shape[1])
        
        # Class distribution
        st.subheader("🎯 Target Variable Distribution")
        # Ensure Classification column is string type
        df['Classification'] = df['Classification'].astype(str)
        class_dist = df['Classification'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_dist.plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'])
        plt.title('IPO Classification Distribution')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(fig)
    """Display data analysis including correlation heatmaps"""
    st.header("📈 Data Analysis & Feature Correlation")
    
    df = load_and_preprocess_data()
    if df is not None:
        with st.spinner("Performing feature engineering for correlation analysis..."):
            X_engineered, _, _, _ = advanced_feature_engineering(df)
        
        # Display correlation heatmaps
        display_correlation_heatmaps(df, X_engineered)
        
        # Additional data insights
        st.header("📋 Dataset Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Original Features", len([
                'P/E', 'Mar Capitalization Rs.Cr.', 'Dividend Yield %',
                'Net Profit of last quarter Rs. Cr.', 'Quarterly Profit Variation %',
                'Quarterly Sales Rs.Cr.', 'Quarterly Sales Variation %',
                'Issue Price (Rs)', 'ROCE %'
            ]))
        with col3:
            st.metric("Engineered Features", X_engineered.shape[1])
        
        # Class distribution
        st.subheader("🎯 Target Variable Distribution")
        class_dist = df['Classification'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        class_dist.plot(kind='bar', color=['#4CAF50', '#FF9800', '#F44336'])
        plt.title('IPO Classification Distribution')
        plt.xlabel('Classification')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        st.pyplot(fig)

def show_model_analysis():
    st.header("📊 Model Performance Analysis")
    
    try:
        with open('model_performance.pkl', 'rb') as f:
            individual_performance = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Get test data for analysis
        df = load_and_preprocess_data()
        if df is not None:
            X, y_encoded, _, original_features = advanced_feature_engineering(df)
            # Recreate the test set to match the training
            _, X_test, _, y_test_encoded = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Convert encoded test labels back to original
            y_test_original = label_encoder.inverse_transform(y_test_encoded)
            
            display_model_performance(individual_performance, y_test_original)
        else:
            st.error("❌ Could not load data for analysis")
            
    except FileNotFoundError:
        st.error("❌ Model performance data not found. Please train the model first in the 'Model Training' section.")

if __name__ == "__main__":
    main()