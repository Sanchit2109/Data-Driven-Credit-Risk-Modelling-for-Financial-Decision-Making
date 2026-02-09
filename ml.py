import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Config 

DATA_PATH = "credit_risk_dataset.csv"
MODEL_PATH = "credit_risk_model.pkl"

TARGET = "loan_status"

# load data 

df = pd.read_csv(DATA_PATH)

print("Dataset Shape (raw):", df.shape)

# ---- Data Cleaning ----

# Remove unrealistic ages (e.g. 144)
df = df[df["person_age"] <= 100]

# Remove extreme employment lengths (> 60 years)
df = df[df["person_emp_length"].fillna(0) <= 60]

# Remove extreme incomes (top 0.1%)
income_cap = df["person_income"].quantile(0.999)
df = df[df["person_income"] <= income_cap]

print("Dataset Shape (cleaned):", df.shape)
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

# ---- Model Comparison ----

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
}

print("\n===== MODEL COMPARISON =====")
for name, mdl in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", mdl)])
    pipe.fit(X_train, y_train)
    score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"  {name}: ROC-AUC = {score:.4f}")

# ---- Hyperparameter Tuning (Random Forest) ----

print("\nRunning GridSearchCV on RandomForest...")

param_grid = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [10, 12, 15],
    "model__min_samples_split": [2, 5],
}

base_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1))
])

grid = GridSearchCV(
    base_pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=0
)

grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

pipeline = grid.best_estimator_

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

plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: confusion_matrix.png")

# roc_curve 

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
print("Saved: roc_curve.png")

# calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("Predicted PD")
plt.ylabel("Observed Default Rate")
plt.title("Calibration Curve")
plt.savefig("calibration_curve.png", dpi=150, bbox_inches="tight")
print("Saved: calibration_curve.png")

# PD(Probability of Default) Distribution

plt.figure()
plt.hist(y_prob, bins=30)
plt.xlabel("Predicted PD")
plt.ylabel("Count")
plt.title("Distribution of Probability of Default")
plt.savefig("pd_distribution.png", dpi=150, bbox_inches="tight")
print("Saved: pd_distribution.png")

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
plt.savefig("feature_importances.png", dpi=150, bbox_inches="tight")
print("Saved: feature_importances.png")

joblib.dump(pipeline, MODEL_PATH)
print("\nModel saved as:", MODEL_PATH)

# ---- SHAP Explainability ----

try:
    import shap

    # Get transformed data for SHAP (use sample of 200 for speed)
    X_test_sample = X_test.iloc[:200]
    X_sample_transformed = pipeline.named_steps["preprocess"].transform(X_test_sample)
    rf_model = pipeline.named_steps["model"]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample_transformed)

    # Handle both old (list) and new (3D array) SHAP output formats
    if isinstance(shap_values, list):
        sv_default = shap_values[1]  # class 1 = default
        base_val = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        sv_default = shap_values[:, :, 1]
        base_val = explainer.expected_value[1]
    else:
        sv_default = shap_values
        base_val = explainer.expected_value

    # Convert sparse matrix to dense if needed
    if hasattr(X_sample_transformed, "toarray"):
        X_dense = X_sample_transformed.toarray()
    else:
        X_dense = np.array(X_sample_transformed)

    # Global feature importance (SHAP)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        sv_default,
        X_dense,
        feature_names=list(feature_names),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")

    # Single prediction explanation
    sample_idx = 0
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv_default[sample_idx],
            base_values=base_val,
            data=X_dense[sample_idx],
            feature_names=list(feature_names)
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_waterfall_sample.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_waterfall_sample.png")

except ImportError:
    print("\nSHAP not installed. Run: pip install shap")

# ---- Risk Prediction Example ----

def risk_bucket(pd_value):
    if pd_value < 0.10:
        return "Low"
    elif pd_value < 0.25:
        return "Medium"
    else:
        return "High"

sample = X_test.iloc[[0]]

pd_val = pipeline.predict_proba(sample)[0][1]

print("\n===== SAMPLE APPLICANT =====")
print(sample)

print("\nPD:", round(pd_val, 3))
print("Risk Level:", risk_bucket(pd_val))
