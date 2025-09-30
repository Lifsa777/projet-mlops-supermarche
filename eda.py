import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from data_processor import DataProcessor
import os

os.makedirs('outputs', exist_ok=True)

# Chargement
df_traited= Path("Data") / "donnees_traitees.csv"
df_traited = pd.read_csv(df_traited)
df_merged = pd.read_csv('data/merged_data.csv')  # Après prépa
print("Traitée shape:", df_traited.shape, "Merged shape:", df_merged.shape)

# Stats descriptives (merged pour training)
print("\n=== STATS MERGED (POUR TRAINING) ===")
print(df_merged.describe())

processor = DataProcessor()
X, y = processor.fit_transform(df_merged)

# Graphs (sur merged)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.histplot(y, kde=True, ax=axes[0,0]).set_title('Distribution Surplus (Merged)')
sns.boxplot(x='Product line', y='surplus', data=df_merged, ax=axes[0,1]).set_title('Surplus par Product Line')
sns.boxplot(x='Branch', y='surplus', data=df_merged, ax=axes[0,2]).set_title('Surplus par Branch')
sns.scatterplot(x='jours_avant_peremption', y='surplus', data=df_merged, ax=axes[1,0]).set_title('Surplus vs Jours')
sns.scatterplot(x='Unit price', y='surplus', data=df_merged, ax=axes[1,1]).set_title('Surplus vs Prix')
sns.boxplot(x='jour_semaine', y='surplus', data=df_merged, ax=axes[1,2]).set_title('Surplus par Jour')
plt.tight_layout()
plt.savefig('outputs/eda_merged.png')
plt.show()

# Corrélations
num_cols = ['Unit price', 'Rating', 'stock_initial', 'jours_avant_peremption', 'jour_semaine', 'mois', 'surplus']
corr = df_merged[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Corrélations (Merged)')
plt.savefig('outputs/corr_merged.png')
plt.show()

# Outliers (merged)
print("\n=== OUTLIERS (MERGED) ===")
for col in num_cols:
    Q1, Q3 = df_merged[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df_merged[col] < Q1 - 1.5*IQR) | (df_merged[col] > Q3 + 1.5*IQR)).sum()
    print(f"{col}: {outliers} outliers")

# Comparaison datasets
print("\n=== COMPARAISON DATASETS ===")
print("Moyenne surplus - Traitée:", df_traited['surplus'].mean())
print("Moyenne surplus - Merged:", df_merged['surplus'].mean())

print("EDA terminé.")