import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/features_v1.csv")

def evaluate_model(y_true, y_pred_class, y_pred_proba):
    return{
        "accuracy": accuracy_score(y_true, y_pred_class),
        "log_loss": log_loss(y_true, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred_class)
    }

# Split data into training and evaluation sets based on date
df_train = df[df['Date'] < '2023-08-01'].copy()
df_train_calib = df[(df['Date'] > '2023-08-01') & (df['Date'] < '2024-08-01')].copy()
df_eval = df[df['Date'] >= '2024-08-01'].copy()

# Create a df_eval_meta for storing match metadata
df_eval_meta = df_eval[['Date','HomeTeam','AwayTeam','FTR']].copy()

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
model_1 = LogisticRegression(max_iter=1000)
model_1.fit(X_train_log, y_train)
pred_prob_lr = model_1.predict_proba(X_eval_log)
pred_class_lr = model_1.predict(X_eval_log)

# Slicing arrays to get A,D,H probabilies for LR
A_prob_lr = pred_prob_lr[:,0]
D_prob_lr = pred_prob_lr[:,1]
H_prob_lr = pred_prob_lr[:,2]
print(A_prob_lr.min(),A_prob_lr.mean(),A_prob_lr.max(),'\n')
print(D_prob_lr.min(),D_prob_lr.mean(),D_prob_lr.max(),'\n')
print(H_prob_lr.min(),H_prob_lr.mean(),H_prob_lr.max(),'\n')

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
pred_prob_xgb = model_2.predict_proba(X_eval_tree)
pred_class_xgb = model_2.predict(X_eval_tree)

# Draw probability stats for XGB
draw_idx = list(le.classes_).index('D')
draw_prob_xgb = pred_prob_xgb[:, draw_idx]
print(draw_prob_xgb.shape)
print("Draw prob stats (XGB):\n")
print("Min :\n", draw_prob_xgb.min())
print("Mean:\n", draw_prob_xgb.mean())
print("Max :\n", draw_prob_xgb.max())

# Calibration
calibrated_clf = CalibratedClassifierCV(
    estimator=FrozenEstimator(model_2),
    method='sigmoid',
)
calibrated_clf.fit(X_train_calib,y_train_calib_encoded)
pred_prob_calibrated = calibrated_clf.predict_proba(X_eval_tree)

pred_class_calibrated = []

def draw_decision(p_home, p_draw, p_away):
    draw_threshold = 0.20
    balance_threshold = 0.20
    margin = 0.15
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
    pred = draw_decision(p_home,p_draw,p_away)
    pred_class_calibrated.append(pred)

pred_class_calibrated = np.array(pred_class_calibrated)
draw_prob_xgb_calibrated = pred_prob_calibrated[:, draw_index]

#Creating a baseline version A where home team wins everytime
y_pred_class_baseline = np.full(len(y_eval),"H",dtype=str)
y_pred_prob_baseline = np.full((len(y_eval),3),[0,0,1])

# Evaluate models
metrics_lr = evaluate_model(y_eval, pred_class_lr, pred_prob_lr)
print('Model Score Logistic Regression:\n',metrics_lr,'\n')
metrics_xgb = evaluate_model(y_eval_encoded,pred_class_xgb, pred_prob_xgb)
print('Model Score XGBoost (Uncalibrated):\n',metrics_xgb)
metrics_xgb_calib = evaluate_model(y_eval_decoded, pred_class_calibrated, pred_prob_calibrated)
print('Model Score XGBoost (Calibrated):\n',metrics_xgb_calib,'\n')
metrics_baseline = evaluate_model(y_eval, y_pred_class_baseline, y_pred_prob_baseline)
print('Baseline Score:\n',metrics_baseline)

# --- CM plot for XGB Calibrated --- -
cm = metrics_xgb_calib['confusion_matrix']
labels = ["Home", "Draw", "Away"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap='Blues',values_format='d')
plt.title("XGBoost(Calibrated) – Confusion Matrix")
plt.savefig("C:/Users/Pratik/DS/premier-league-ml/reports/figures/CM_XGBoost_Calibrated.png", dpi=300, bbox_inches="tight")
plt.show()

# ===== Reliability Curves =====
from sklearn.calibration import calibration_curve

# Reliability curve for Draws - Logistic Regression
prob_true_lr, prob_pred_lr = calibration_curve(
    y_eval == 'D',  #binary ground truth for Draws
    D_prob_lr,      #predicted probabilities for Draws from Logistic Regression
    n_bins=10,
    strategy='uniform'
    )

# Reliability curve plot for Draws - XGBoost
prob_true_xgb, pred_prob_xgb = calibration_curve(
    y_eval == 'D',   #binary ground truth for Draws
    draw_prob_xgb,   #predicted probabilities for Draws from XGBoost
    n_bins=10,
    strategy='uniform'
)

# Reliability curve plot for Draws - XGBoost Calibrated
prob_true_xgb_calib, pred_prob_xgb_calib = calibration_curve(
    y_eval == 'D',                          #binary ground truth for Draws  
    draw_prob_xgb_calibrated,    #predicted probabilities for Draws from Calibrated XGBoost
    n_bins=10,
    strategy='uniform'
)


# Plotting Reliability Curve for Draws - Logistic Regression
plot_title = 'Reliability Curve - Draws Comparison'
plt.figure(figsize=(8,6))
plt.plot(prob_pred_lr, prob_true_lr, marker='o', label='Logistic Regression', color='blue')
plt.plot(pred_prob_xgb, prob_true_xgb, marker='o', label='XGBoost', color='green')
plt.plot(pred_prob_xgb_calib, prob_true_xgb_calib, marker='o', label='XGBoost Calibrated', color='red')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed draw frequency")
plt.legend()
plt.grid(True)
plt.savefig("C:/Users/Pratik/DS/premier-league-ml/reports/figures/reliability_curve_draws_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# ---- ECE -----
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    y_true: binary array (1 if draw, 0 otherwise)
    y_prob: predicted probability for draw
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return ece
y_draw_true = (y_eval == 'D').astype(int)

ece_lr = expected_calibration_error(y_draw_true, D_prob_lr)
ece_xgb = expected_calibration_error(y_draw_true, draw_prob_xgb)
ece_xgb_cal = expected_calibration_error(
    y_draw_true, draw_prob_xgb_calibrated
)
print("ECE (Draw):")
print("Logistic Regression:", ece_lr)
print("XGBoost:", ece_xgb)
print("XGBoost Calibrated:", ece_xgb_cal)

# ----- Brier Score -----
from sklearn.metrics import brier_score_loss

brier_lr = brier_score_loss(y_draw_true, D_prob_lr)
brier_xgb = brier_score_loss(y_draw_true, draw_prob_xgb)
brier_xgb_cal = brier_score_loss(
    y_draw_true, draw_prob_xgb_calibrated
)

print("Brier score (Draw):")
print("Logistic Regression:", brier_lr)
print("XGBoost:", brier_xgb)
print("XGBoost Calibrated:", brier_xgb_cal)


# ---- Summary Stats -----
summary = pd.DataFrame({
    "Model": [
        "Baseline (Always Home)",
        "Logistic Regression",
        "XGBoost",
        "XGBoost Calibrated"
    ],
    "Accuracy": [
        metrics_baseline["accuracy"],
        metrics_lr["accuracy"],
        metrics_xgb["accuracy"],
        metrics_xgb_calib["accuracy"]
    ],
    "Log Loss": [
        metrics_baseline["log_loss"],
        metrics_lr["log_loss"],
        metrics_xgb["log_loss"],
        metrics_xgb_calib["log_loss"]
    ],
    "Draw Recall": [
        metrics_baseline["confusion_matrix"][1,1] / metrics_baseline["confusion_matrix"][1].sum(),
        metrics_lr["confusion_matrix"][1,1] / metrics_lr["confusion_matrix"][1].sum(),
        metrics_xgb["confusion_matrix"][1,1] / metrics_xgb["confusion_matrix"][1].sum(),
        metrics_xgb_calib["confusion_matrix"][1,1] / metrics_xgb_calib["confusion_matrix"][1].sum() if metrics_xgb_calib["confusion_matrix"][1].sum() > 0 else 0.0
    ],
    "ECE (Draw)": [
        np.nan,
        ece_lr,
        ece_xgb,
        ece_xgb_cal
    ],
    "Brier (Draw)": [
        np.nan,
        brier_lr,
        brier_xgb,
        brier_xgb_cal
    ]
})

print(summary,'\n')
summary.to_csv('C:/Users/Pratik/DS/premier-league-ml/reports/tables/model_metrics_summary.csv',index=False)

# Match Level Predictions and Probabilities Output to CSV for the Calibrated XGBoost Model
match_level_probs = df_eval_meta.copy()

match_level_probs["P_Away"] = pred_prob_calibrated[:, away_index]
match_level_probs["P_Draw"] = pred_prob_calibrated[:, draw_index]
match_level_probs["P_Home"] = pred_prob_calibrated[:, home_index]
match_level_probs["Predicted"] = pred_class_calibrated

output_df = match_level_probs[[
    "Date", "HomeTeam", "AwayTeam",
    "P_Home", "P_Draw", "P_Away",
    "Predicted", "FTR"
]]

print(output_df.head())
output_df.to_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/match_probabilities_xgb_calibrated.csv", index=False)
