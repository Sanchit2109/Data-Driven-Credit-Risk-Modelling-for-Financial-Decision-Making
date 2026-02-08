import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier

# Config 

DATA_PATH = "credit_risk_dataset.csv"
MODEL_PATH = "credit_risk_model.pkl"

TARGET = "loan_status"

# load data 

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

# Features and target 

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Train test split 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Columns type 

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("\nCategorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

# Preprocessing

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# Model 

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# Train 

pipeline.fit(X_train, y_train)

# Evaluation

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC:", auc)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))

# Confusion matrix 

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1])
plt.yticks([0, 1])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# roc_curve 

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Predicted PD")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Curve")
plt.show()

# PD(Probability of Default) Distribution

plt.figure()
plt.hist(y_prob, bins=30)
plt.xlabel("Predicted PD")
plt.ylabel("Count")
plt.title("Distribution of Probability of Default")
plt.show()

# Feature 
feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False).head(15)

plt.figure()
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances")
plt.show()

joblib.dump(pipeline, MODEL_PATH)
print("\nModel saved as:", MODEL_PATH)

# Example of prediction model 

def risk_bucket(pd):
    if pd < 0.10:
        return "Low"
    elif pd < 0.25:
        return "Medium"
    else:
        return "High"

sample = X_test.iloc[[0]]

pd_val = pipeline.predict_proba(sample)[0][1]

print("\n===== SAMPLE APPLICANT =====")
print(sample)

print("\nPD:", round(pd_val, 3))
print("Risk Level:", risk_bucket(pd_val))
