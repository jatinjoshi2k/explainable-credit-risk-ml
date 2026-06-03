import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# 1. Load cleaned dataset
df = pd.read_csv("credit_risk_cleaned.csv")

# 2. Separate features and target
X = df.drop(columns=["loan_status"])
y = df["loan_status"]

print("=" * 50)
print("DATA SPLIT")
print("=" * 50)
print(f"Total samples  : {len(df)}")
print(f"Features       : {X.shape[1]}")
print(f"Default rate   : {y.mean():.2%}  (1=default, 0=paid)")

# 3. Train / test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nTraining set   : {X_train.shape[0]} rows")
print(f"Test set       : {X_test.shape[0]} rows")

# 4. Train XGBoost classifier
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)
print("Training complete.")

# 5. Evaluate on test set
y_pred = model.predict(X_test)

print("\n" + "=" * 50)
print("MODEL EVALUATION ON TEST SET")
print("=" * 50)
print(f"Accuracy Score : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Paid (0)", "Default (1)"]))

# 6. Save the model
model.save_model("loan_model.json")
print("Model saved: loan_model.json")
