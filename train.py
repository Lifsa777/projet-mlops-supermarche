import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import warnings
from data_processor import DataProcessor
import os

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# Chargement merged
df = pd.read_csv('data/merged_data.csv')
print("Merged chargé. Shape:", df.shape)

# Préprocessing
processor = DataProcessor()
X, y = processor.fit_transform(df)
processor.save('outputs/preprocessor.pkl')

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles avec GridSearch élargi
models_to_test = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]}
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    },
    'LinearRegression': {
        'model': LinearRegression(),
        'param_grid': {}  # Pas de grid
    }
}

results = {}
best_model = None
best_mse = np.inf
best_params = {}

for name, config in models_to_test.items():
    print(f"\n--- {name} ---")
    estimator = config['model']
    param_grid = config['param_grid']
    
    if param_grid:
        grid = GridSearchCV(estimator, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        curr_params = grid.best_params_
        print(f"Meilleurs params: {curr_params}")
    else:
        model = estimator.fit(X_train, y_train)
        curr_params = {}
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2, 'params': curr_params}
    
    cv_mse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()
    print(f"MSE Test: {mse:.4f} | R2: {r2:.4f} | CV MSE: {cv_mse:.4f}")
    
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_params = curr_params

# Sauvegarde
joblib.dump(best_model, 'outputs/best_model.pkl')
with open('outputs/metrics.json', 'w') as f:
    json.dump(results, f, indent=4, default=str)

print("\n=== SYNTHÈSE ===")
for name, metrics in results.items():
    print(f"{name}: MSE={metrics['MSE']:.4f}, R²={metrics['R2']:.4f}, Params={metrics['params']}")

best_name = min(results, key=lambda k: results[k]['MSE'])
print(f"\nMeilleur: {best_name} (MSE: {best_mse:.4f})")
print("Modèle sauvé dans outputs/best_model.pkl")