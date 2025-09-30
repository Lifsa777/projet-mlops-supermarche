import pandas as pd
import numpy as np
from pathlib import Path
import os

os.makedirs('data', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# Chargement
df_original = Path("Data") / "SuperMarket Analysis.csv"
df_traited= Path("Data") / "donnees_traitees.csv"
df_traited = pd.read_csv(df_traited)
df_original = pd.read_csv(df_original)

print("Shapes: Traitée=", df_traited.shape, "Originaux=", df_original.shape)

# Labellisation/Features pour originaux
df_original['Date'] = pd.to_datetime(df_original['Date'])
df_original['jour_semaine'] = df_original['Date'].dt.dayofweek  # 0=Lun
df_original['mois'] = df_original['Date'].dt.month
df_original['stock_initial'] = np.random.randint(10, 50, len(df_original))  # Synthétique
df_original['jours_avant_peremption'] = np.random.randint(1, 60, len(df_original))  # Synthétique
df_original['surplus'] = df_original['stock_initial'] - df_original['Quantity']  # Approx labellisation
df_original['surplus'] = np.maximum(0, df_original['surplus'])  # >=0

# Select features communes
features = ['Branch', 'City', 'Customer type', 'Product line', 'Unit price', 'Rating',
            'stock_initial', 'jours_avant_peremption', 'jour_semaine', 'mois']
df_original = df_original[features + ['surplus']].dropna()

# Merge (concat pour plus de data)
df_merged = pd.concat([df_traited[features + ['surplus']], df_original], ignore_index=True)
df_merged.to_csv('data/merged_data.csv', index=False)
print("Merged sauvé: data/merged_data.csv, shape=", df_merged.shape)
print(df_merged.head())