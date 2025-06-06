import pandas as pd
import joblib # Library for saving and loading Python objects efficiently
import os # For checking if the model file exists

print("--- Running the Saved AI Model for Predictions ---")

# Define the filename of your saved model
model_filename = 'churn_predictor_model.joblib'

# --- Step 1.1: Check if the model file exists ---
if not os.path.exists(model_filename):
    print(f"Error: Model file '{model_filename}' not found.")
    print("Please make sure you've run 'train_churn_model.py' first to generate the model.")
    exit() # Exit the script if the model isn't found

# --- Step 1.2: Load the Trained Model ---
print(f"Loading the trained model from '{model_filename}'...")
try:
    loaded_model = joblib.load(model_filename)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("The model file might be corrupted or was not saved correctly.")
    exit()

# Verify that the loaded model has the parameters (optional, but good for confirmation)
print("\n--- Loaded Model Parameters (Confirmation) ---")
print(f"Coefficients (Weights): {loaded_model.coef_[0]}")
print(f"Intercept (Bias): {loaded_model.intercept_[0]}")
print("--- Model is ready for predictions! ---\n")

# --- Step 1.3: Prepare New Data for Prediction ---
print("Preparing new, unseen customer data for prediction:")

# Create some new hypothetical customer data
# Note: The column names ('age', 'monthly_bill') must match those used during training.
new_customers_data = {
    'age': [35, 60, 28, 50, 42],
    'monthly_bill': [75.0, 110.5, 45.2, 5000.0, 130.0]
}
new_customers_df = pd.DataFrame(new_customers_data)

print("New customer data to predict:")
print(new_customers_df)
print("\n")

# --- Step 1.4: Make Predictions with the Loaded Model ---
print("Making predictions...")

# Use the loaded model's .predict() method to get the predicted class (0 for no churn, 1 for churn)
predictions = loaded_model.predict(new_customers_df)

# Use .predict_proba() to get the probabilities for each class (e.g., [prob_no_churn, prob_churn])
probabilities = loaded_model.predict_proba(new_customers_df)

print("--- Prediction Results ---")
for i, (age, bill) in enumerate(zip(new_customers_df['age'], new_customers_df['monthly_bill'])):
    # Determine the churn status based on the prediction (0 or 1)
    churn_status = "Will Churn" if predictions[i] == 1 else "Will NOT Churn"
    # Extract probabilities for 'no churn' and 'churn'
    prob_no_churn = probabilities[i][0]
    prob_churn = probabilities[i][1]

    print(f"Customer {i+1}: (Age: {age}, Bill: ${bill:.2f})")
    print(f"  -> Predicted Churn Status: {churn_status}")
    print(f"  -> Probability (No Churn): {prob_no_churn:.4f}")
    print(f"  -> Probability (Churn): {prob_churn:.4f}\n")

print("--- Prediction process complete. ---")
