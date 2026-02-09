# Data-Driven Credit Risk Modelling for Financial Decision-Making

This project uses machine learning to predict whether a loan applicant is likely to default. It calculates a Probability of Default (PD) for each borrower and groups them into Low, Medium, or High risk.

## Objectives

- Predict loan default using ML
- Estimate Probability of Default for each borrower
- Classify applicants into risk groups (Low / Medium / High)
- Provide a data-driven alternative to traditional credit scoring

## Dataset

32,581 records with 12 columns. Key features include:

- Age, Income, Employment Length
- Home Ownership, Loan Purpose, Loan Grade
- Loan Amount, Interest Rate, Loan-to-Income Ratio
- Credit History Length, Previous Default History

Target: `loan_status` (0 = No Default, 1 = Default)

## Models Used

- **Logistic Regression** — baseline
- **Random Forest** — ensemble method, handles complex patterns
- **Gradient Boosting** — sequential error correction, strong performance

## Workflow

1. Load and clean data
2. Exploratory analysis
3. Feature engineering and preprocessing
4. Train-test split (80/20, stratified)
5. Train models
6. Evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
7. Calculate PD and assign risk buckets

## Risk Buckets

| PD | Risk |
|----|------|
| < 20% | Low |
| 20–50% | Medium |
| > 50% | High |

## Results (Random Forest)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.93 |
| Accuracy | 93% |
| Precision (default) | 0.91 |
| Recall (default) | 0.73 |
| F1 (default) | 0.81 |

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, SHAP, Streamlit

## How to Run

```bash
pip install -r requirements.txt
python ml.py          # Train model and generate plots
streamlit run app.py  # Launch web app
```

Outputs:
- Trained model: `credit_risk_model.pkl`
- Plots: confusion matrix, ROC curve, calibration curve, PD distribution, feature importances, SHAP summary, SHAP waterfall
- Web app at `http://localhost:8501`

## Future Work

- Real-time scoring dashboard
- Integration with alternative financial data
- Model retraining pipeline

## Team

Sanchit Panwar, Aayush Kumar, Saksham Raj

Guide: Dr. Surbhi Saraswat

