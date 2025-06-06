import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib # Library for saving and loading Python objects efficiently

print("--- Phase 1: Training a Simple Model ---")
print("Step 1.1: Generating Synthetic Data for Churn Prediction...")
# Generate Synthetic Data
np.random.seed(42) # for reproducibility
data_size = 500
ages = np.random.randint(20, 70, size=data_size)
monthly_bills = np.random.uniform(30, 150, size=data_size)

# Create a 'churn' column based on a simple rule with some randomness
# Older and higher bill -> higher churn probability
churn_prob = (ages * 0.01) + (monthly_bills * 0.005) - 1.0 + np.random.normal(0, 0.2, size=data_size)
churn = (churn_prob > 0.5).astype(int) # Convert probabilities to 0 or 1 (churn/no churn)

df = pd.DataFrame({
    'age': ages,
    'monthly_bill': monthly_bills,
    'churn': churn
})

X = df[['age', 'monthly_bill']]
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 1.2: Training the Logistic Regression Model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete!")

# Display the learned parameters (weights and bias) - This IS the "trained model's knowledge"
print("\n--- Learned Model Parameters ---")
print(f"Coefficients (Weights) for features: {model.coef_[0]}")
print(f"  - Weight for 'age': {model.coef_[0][0]:.4f}")
print(f"  - Weight for 'monthly_bill': {model.coef_[0][1]:.4f}")
print(f"Intercept (Bias): {model.intercept_[0]:.4f}")
print("---\n")

print("Step 1.3: Saving the Trained Model...")
# This saves the 'model' object, including all its learned parameters.
model_filename = 'churn_predictor_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved locally as '{model_filename}'")
print("\n--- Phase 1 Complete ---")
print("This 'churn_predictor_model.joblib' file now contains the 'trained AI model' for churn prediction.")
print("It's a collection of numbers (weights & bias) that represent the model's learned relationship.")
