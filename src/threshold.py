import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
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

# Encode target labels for XGBoost
le = LabelEncoder()
le.fit(y_train)
y_encoded = le.transform(y_train)
y_eval_encoded = le.transform(y_eval)
y_train_calib_encoded = le.transform(y_train_calib)
y_eval_decoded = le.inverse_transform(y_eval_encoded)

# Adding sample weights
sample_weights = compute_sample_weight(class_weight = {
    0:1, # Away
    1:1.5, # Draw
    2:1} # Home
    , y = y_encoded)

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

pred_class_calibrated = []

def draw_decision_calibrated(p_home, p_draw, p_away):
    draw_threshold = 0.28
    balance_threshold = 0.15
    margin = 0.10

    if (
        p_draw >= draw_threshold
        and abs(p_home - p_away) <= balance_threshold
        and p_draw >= max(p_home, p_away) - margin
    ):
        return 'D'
    else:
        return 'H' if p_home > p_away else 'A'

draw_index = list(le.classes_).index('D')
home_index = list(le.classes_).index('H')
away_index = list(le.classes_).index('A')
for probs in pred_prob_calibrated:
    p_draw = probs[draw_index]
    p_home = probs[home_index]
    p_away = probs[away_index]
    pred = draw_decision_calibrated(p_home,p_draw,p_away)
    pred_class_calibrated.append(pred)

pred_class_calibrated = np.array(pred_class_calibrated)

def evaluate_model(y_true, y_pred_class, y_pred_proba):
    return{
        "accuracy": accuracy_score(y_true, y_pred_class),
        "log_loss": log_loss(y_true, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred_class)
    }

def draw_decision(p_home, p_draw, p_away,
                  draw_threshold,
                  balance_threshold,
                  margin):
    if (
        p_draw >= draw_threshold
        and abs(p_home - p_away) <= balance_threshold
        and p_draw >= max(p_home, p_away) - margin
    ):
        return 'D'
    return 'H' if p_home > p_away else 'A'


draw_thresholds = [0.20, 0.25, 0.30, 0.35]
balance_thresholds = [0.10, 0.15, 0.20]
margins = [0.05, 0.10, 0.15]

results = []

for dt in draw_thresholds:
    for bt in balance_thresholds:
        for m in margins:

            preds = []
            for p in pred_prob_calibrated:
                preds.append(
                    draw_decision(
                        p_home=p[home_index],
                        p_draw=p[draw_index],
                        p_away=p[away_index],
                        draw_threshold=dt,
                        balance_threshold=bt,
                        margin=m
                    )
                )

            preds = np.array(preds)

            metrics = evaluate_model(
                y_eval,
                preds,
                pred_prob_calibrated
            )

            results.append({
                "draw_threshold": dt,
                "balance_threshold": bt,
                "margin": m,
                "accuracy": metrics["accuracy"],
                "log_loss": metrics["log_loss"],
                "draw_recall": (
                    (preds == 'D') & (y_eval == 'D')
                ).sum() / (y_eval == 'D').sum()
            })

results_df = pd.DataFrame(results)

results_df.sort_values(
    by=["log_loss", "draw_recall"],
    ascending=[True, False]
).head(10)