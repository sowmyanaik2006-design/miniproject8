import os
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. Create sample electricity dataset if needed
# --------------------------------------------------
DATA_FILE = "electric_data.csv"

if not os.path.exists(DATA_FILE):
    np.random.seed(42)
    n = 500

    units = np.random.randint(30, 601, n)
    occupants = np.random.randint(1, 8, n)
    days = np.random.randint(28, 32, n)
    daily_hours = np.random.uniform(2, 16, n)

    # Illustrative slab-based electricity tariff
    energy_charge = (
        np.minimum(units, 100) * 3
        + np.minimum(np.maximum(units - 100, 0), 100) * 5
        + np.minimum(np.maximum(units - 200, 0), 200) * 7
        + np.maximum(units - 400, 0) * 9
    )

    bill = (
        energy_charge
        + 50
        + occupants * 10
        + np.random.normal(0, 15, n)
    )
    bill = np.maximum(bill, 0)

    data = pd.DataFrame({
        "Units_Consumed": units,
        "Occupants": occupants,
        "Billing_Days": days,
        "Daily_Usage_Hours": daily_hours.round(2),
        "Electricity_Bill": bill.round(2)
    })

    data.to_csv(DATA_FILE, index=False)
    print("Sample dataset created:", DATA_FILE)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
data = pd.read_csv(DATA_FILE)

required_columns = [
    "Units_Consumed",
    "Occupants",
    "Billing_Days",
    "Daily_Usage_Hours",
    "Electricity_Bill"
]

if not all(column in data.columns for column in required_columns):
    raise ValueError(
        "CSV must contain: " + ", ".join(required_columns)
    )

data = data[required_columns].dropna()

X = data[
    [
        "Units_Consumed",
        "Occupants",
        "Billing_Days",
        "Daily_Usage_Hours"
    ]
]

y = data["Electricity_Bill"]

if len(data) < 10:
    raise ValueError("Please provide at least 10 valid rows.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 3. Configure MLflow
# --------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Electricity_Bill_Prediction")

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------
models = {
    "Linear_Regression": LinearRegression(),

    "Random_Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "Gradient_Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []
best_r2 = float("-inf")
best_model = None
best_model_name = None

# --------------------------------------------------
# 5. Train and track experiments
# --------------------------------------------------
for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_name", name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("R2_Score", float(r2))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        print(f"\nModel: {name}")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name

# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="MAE", ascending=True
)

results_df.to_csv("model_comparison.csv", index=False)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. Save the best model
# --------------------------------------------------
with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest Model:", best_model_name)
print(f"Best Model R2 Score: {best_r2:.4f}")
print("Saved: best_model.pkl")
print("Saved: model_comparison.csv")
print("MLflow tracking completed successfully!")
