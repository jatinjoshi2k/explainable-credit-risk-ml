import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# 1. Load cleaned dataset and reproduce the same test split
df = pd.read_csv("credit_risk_cleaned.csv")
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

_, X_test, _, _ = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Load the saved model
model = XGBClassifier()
model.load_model("loan_model.json")

# 3. Compute SHAP values using TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 4. Generate and save the beeswarm summary plot
fig, ax = plt.subplots(figsize=(12, 9))

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="dot",
    max_display=20,
    show=False,
    plot_size=None,
)

plt.title("SHAP Feature Importance — Loan Default Prediction", fontsize=14, pad=14)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: shap_summary.png")
