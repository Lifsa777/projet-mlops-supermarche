import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')
print("📦 Début du script d'entraînement...")

# --------------------
# 1) Chargement des données
# --------------------
DATA = Path("Data") / "SuperMarket Analysis.csv"
saved_model = Path("saved_model"); saved_model.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])

# --------------------
# 2) Feature engineering
# --------------------
df["jour_de_semaine"] = df["Date"].dt.dayofweek
df["mois"] = df["Date"].dt.month
df["surplus"] = df["cogs"] - df["Quantity"]
df["surplus"] = df["surplus"].clip(lower=0)

# --------------------
# 3) Sélection des features et de la cible
# --------------------
features = [
    "Branch", "City", "Product line", "Unit price", "Quantity",
    "Payment", "Rating", "jour_de_semaine", "mois"
]
target = "surplus"

X = df[features].copy()
y = df[target].values

cat_cols = ["Branch", "City", "Product line", "Payment"]
num_cols = ["Unit price", "Quantity", "Rating", "jour_de_semaine", "mois"]

# --------------------
# 4) Pipeline de prétraitement + modèle
# --------------------
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False))
    ]), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("rf", model)
])

# --------------------
# 5) Split temporel (TimeSeriesSplit)
# --------------------
df_sorted = df.sort_values("Date").reset_index(drop=True)
X_sorted = X.loc[df_sorted.index]
y_sorted = y[df_sorted.index]

tscv = TimeSeriesSplit(n_splits=5)
tr_idx, te_idx = list(tscv.split(X_sorted))[-1]

X_train, X_test = X_sorted.iloc[tr_idx], X_sorted.iloc[te_idx]
y_train, y_test = y_sorted[tr_idx], y_sorted[te_idx]

# --------------------
# 6) Entraînement + Évaluation
# --------------------
print("🚀 Entraînement du modèle RandomForest...")
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

mae  = mean_absolute_error(y_test, pred)
rmse = mean_squared_error(y_test, pred) ** 0.5
r2   = r2_score(y_test, pred)

print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")

# --------------------
# 7) Sauvegarde du modèle et des métriques
# --------------------
model_path = saved_model/ "surplus_rf.joblib"
joblib.dump(pipe, model_path)
print(f"✅ Modèle sauvegardé → {model_path}")

metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "n_test": int(len(y_test))}
with open(saved_model / "metrics_surplus_rf.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("📁 Métriques sauvegardées →", saved_model/ "metrics_surplus_rf.json")