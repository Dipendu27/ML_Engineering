import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

print("--- Day 16: Advanced XGBoost & Hyperparameter Tuning ---\n")

# 1. Generate Synthetic Patient Treatment Cost Data
np.random.seed(42)
n_patients = 2000

ages = np.random.randint(18, 65, n_patients)
bmis = np.random.normal(28, 6, n_patients)
# 1 = Smoker, 0 = Non-Smoker
smokers = np.random.choice([0, 1], size=n_patients, p=[0.8, 0.2]) 

# Logic: Base cost + age factor + slight BMI factor + MASSIVE Smoker penalty
# We add non-linear noise because healthcare costs are unpredictable
costs = 2000 + (ages * 50) + (bmis * 15) + (smokers * 15000) + np.random.normal(0, 2500, n_patients)

df = pd.DataFrame({'Age': ages, 'BMI': bmis, 'Smoker': smokers, 'Treatment_Cost': costs})

X = df.drop('Treatment_Cost', axis=1)
y = df['Treatment_Cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 2. The Baseline Model (Default Settings)
# ---------------------------------------------------------
print("🚀 Training Baseline XGBoost Model (Default Settings)...")
baseline_model = xgb.XGBRegressor(random_state=42)
baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
print(f"📉 Baseline Error: Off by ${baseline_mae:,.2f} per patient on average.\n")


# ---------------------------------------------------------
# 3. Hyperparameter Tuning (The Search Grid)
# ---------------------------------------------------------
print("⚙️ Running Randomized Search to find the optimal hyperparameters...")
print("   (This trains dozens of models in the background, please wait 3-5 seconds...)")

# We define a "Grid" of possible settings we want the AI to test
param_grid = {
    'n_estimators': [100, 200, 300],         # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # How fast it learns
    'max_depth': [3, 4, 5, 6],               # Complexity of trees
    'subsample': [0.8, 0.9, 1.0],            # % of data used per tree
    'colsample_bytree': [0.8, 0.9, 1.0]      # % of features used per tree
}

# Setup the Search (n_iter=15 means it will randomly try 15 combinations)
# cv=3 means it double-checks its work 3 times per combination (Cross-Validation)
xgboost_base = xgb.XGBRegressor(random_state=42)
search = RandomizedSearchCV(
    estimator=xgboost_base, 
    param_distributions=param_grid, 
    n_iter=15, 
    scoring='neg_mean_absolute_error', 
    cv=3, 
    verbose=0,
    random_state=42,
    n_jobs=-1 # Uses all cores on your M5 chip!
)

# Run the search!
search.fit(X_train, y_train)

# 4. Extracting the Winner
best_model = search.best_estimator_
tuned_predictions = best_model.predict(X_test)
tuned_mae = mean_absolute_error(y_test, tuned_predictions)

print(f"\n🏆 Search Complete! Best parameters found:")
for param, value in search.best_params_.items():
    print(f"   - {param}: {value}")

print(f"\n📉 Tuned Model Error: Off by ${tuned_mae:,.2f} per patient on average.")
print(f"💰 Financial Impact: Tuning saved ${baseline_mae - tuned_mae:,.2f} of error per prediction!")
