# ---------------------------
# 📊 IPO Model Feature Word Cloud
# ---------------------------

# Install if not already installed:
# pip install wordcloud matplotlib pandas

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------------------
# 1️⃣ Option 1: From your feature names
# -----------------------------------------

# Example: features used in your IPO prediction model
feature_names = wordcloud_words = [
    'IPO', 'Initial', 'Public', 'Offering', 'Boosting', 'Machine Learning',
    'Valuation', 'Growth', 'Performance', 'Investment', 'Profit',
    'Revenue', 'Sales', 'Earnings', 'Dividend', 'ROCE', 'Risk',
    'Feature', 'Engineering', 'Model', 'Ensemble', 'Prediction',
    'Classification', 'Accuracy', 'F1-Score', 'Precision', 'Recall',
    'BalancedRandomForest', 'XGBoost', 'LightGBM', 'CatBoost',
    'GradientBoosting', 'Stacking', 'Voting', 'SMOTE', 'Tomek',
    'Data', 'Preprocessing', 'Scaling', 'Selection', 'Training',
    'Testing', 'Metrics', 'Confidence', 'Probability', 'Threshold',
    'MarketPremium', 'GMP', 'Financial', 'Stability', 'Efficiency',
    'Momentum', 'Interaction', 'Polynomial', 'Regression', 'Logistic',
    'Algorithm', 'Evaluation', 'Recommendation', 'Dashboard',
    'Visualization', 'Results', 'Application', 'Input', 'Output',
    'FeatureSelection', 'Imbalanced', 'Oversampling', 'Undersampling'
]


# Join all feature names into one big string
text = " ".join(feature_names)

# Generate the word cloud
wordcloud = WordCloud(
    width=1000,
    height=600,
    background_color='white',
    colormap='plasma',
    prefer_horizontal=0.9,
    min_font_size=10,
    max_font_size=150
).generate(text)

# Display the word cloud
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("IPO Model Feature Word Cloud", fontsize=18)
plt.show()
