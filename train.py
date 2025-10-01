import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import os

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

print("Démarrage du script d'entraînement final...")

# --- Chargement des Données ---
# On utilise bien le fichier que vous avez préparé.
df = pd.read_csv('data/donnees_traitees.csv') 

# Définition des features (X) et de la target (y)
features = [
    'Branch', 'City', 'Product line', 'Unit price', 'stock_initial',
    'jours_avant_peremption', 'jour_semaine', 'mois'
]
target = 'surplus'

X = df[features]
y = df[target]

print(f"Features sélectionnées: {features}")
print(f"Variable cible: {target}")

# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Taille de l'ensemble d'entraînement: {X_train.shape[0]} échantillons")

# Création du pipeline de prétraitement
numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
print("Pipeline de prétraitement créé.")

# Définition du modèle final avec les meilleurs hyperparamètres trouvés
best_params = {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}

# Création du pipeline complet (prétraitement + modèle) avec les meilleurs paramètres
final_model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', GradientBoostingRegressor(random_state=42, **best_params))])


print("\n--- Entraînement du modèle final sur l'ensemble des données ---")
final_model_pipeline.fit(X, y) # Entraînement sur toutes les données pour le modèle final
print("Modèle final entraîné.")

# Evaluation du modèle
y_pred_final = final_model_pipeline.predict(X_test)
final_mae = mean_absolute_error(y_test, y_pred_final)
final_mse = mean_squared_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(f"\nPerformance du modèle final optimisé sur l'ensemble de test:")
print(f"  Final MAE: {final_mae:.2f}")
print(f"  Final MSE: {final_mse:.2f}")
print(f"  Final R² Score: {final_r2:.4f}")

# Sauvegarde du pipeline complet
model_filename = 'saved_model/surplus_predict_model.joblib'
os.makedirs(os.path.dirname(model_filename), exist_ok=True)
joblib.dump(final_model_pipeline, model_filename)
print(f"Modèle sauvegardé avec succès.")
print("\n--- Script d'entraînement terminé ---")