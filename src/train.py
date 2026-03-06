import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/features_v1.csv")

# Split data into training and evaluation sets based on date
df_train = df[df['Date'] < '2023-08-01'].copy()
df_train_calib = df[(df['Date'] > '2023-08-01') & (df['Date'] < '2024-08-01')].copy()
df_eval = df[df['Date'] >= '2024-08-01'].copy()

# Drop non-feature columns
df_train = df_train.drop(columns=['Date','HomeTeam','AwayTeam'])  
df_train_calib = df_train_calib.drop(columns=['Date','HomeTeam','AwayTeam'])
df_eval = df_eval.drop(columns=['Date','HomeTeam','AwayTeam'])

# Separate features and target variable
X_train_raw = df_train.drop(columns=['FTR'])
y_train = df_train['FTR']

X_train_calib = df_train_calib.drop(columns=['FTR'])
y_train_calib = df_train_calib['FTR']

X_eval_raw = df_eval.drop(columns=['FTR'])
y_eval = df_eval['FTR']

# Prepare data for Logistic Regression (handle missing values)
X_train_log = X_train_raw.fillna(0)
X_eval_log = X_eval_raw.fillna(0)

# Prepare data for XGBoost (keep missing values as is)
X_train_tree = X_train_raw.copy()
X_eval_tree = X_eval_raw.copy()

# Train Logistic Regression model
model_1 = LogisticRegression(
    max_iter=1000,
    class_weight={'H':1, 'A': 1, 'D': 2.5}
    )
model_1.fit(X_train_log, y_train)
pred_prob_lr = model_1.predict_proba(X_eval_log)
pred_class_lr = model_1.predict(X_eval_log)

# Encode target labels for XGBoost
le = LabelEncoder()
le.fit(y_train)
print(f'Classes: {le.classes_}\n')
y_encoded = le.transform(y_train)
y_eval_encoded = le.transform(y_eval)
y_train_calib_encoded = le.transform(y_train_calib)
print(f'Encoded labels: {y_encoded[:10]}\n')
print(f'Encoded eval labels: {y_eval_encoded[:10]}\n')

# Adding sample weights
sample_weights = compute_sample_weight(class_weight = {
    0:1, # Away
    1:1.5, # Draw
    2:1} # Home
    , y = y_encoded
    )

# Train XGBoost model
model_2 = XGBClassifier(
    objective="multi:softprob",
    num_class = 3,
    max_depth = 4,
    learning_rate = 0.05,
    n_estimators = 300,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric='mlogloss',
    random_state=42)
model_2.fit(X_train_tree, y_encoded, sample_weight = sample_weights)

# Calibration
calibrated_clf = CalibratedClassifierCV(
    estimator=model_2,
    method='sigmoid',
    cv='prefit'
)
calibrated_clf.fit(X_train_calib,y_train_calib_encoded)
pred_prob_calibrated = calibrated_clf.predict_proba(X_eval_tree)

if __name__ == "__main__":
    print("Training complete")
