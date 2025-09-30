import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

class DataProcessor:
    def __init__(self):
        self.preprocessor = None
        self.is_fitted = False
    
    def fit_transform(self, df):
        features = ['Branch', 'City', 'Customer type', 'Product line', 
                    'Unit price', 'Rating', 'stock_initial', 'jours_avant_peremption', 
                    'jour_semaine', 'mois']
        target = 'surplus'
        
        X = df[features]
        y = df[target]
        
        categorical_cols = ['Branch', 'City', 'Customer type', 'Product line']
        numerical_cols = ['Unit price', 'Rating', 'stock_initial', 'jours_avant_peremption', 
                          'jour_semaine', 'mois']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),  # Standardisation
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)  # Encoding
            ]
        )
        
        X_processed = self.preprocessor.fit_transform(X)
        self.is_fitted = True
        
        X_df = pd.DataFrame(X_processed, columns=[f'feat_{i}' for i in range(X_processed.shape[1])])
        return X_df, y
    
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Fittez d'abord!")
        
        features = ['Branch', 'City', 'Customer type', 'Product line', 
                    'Unit price', 'Rating', 'stock_initial', 'jours_avant_peremption', 
                    'jour_semaine', 'mois']
        X = df[features]
        
        X_processed = self.preprocessor.transform(X)
        X_df = pd.DataFrame(X_processed, columns=[f'feat_{i}' for i in range(X_processed.shape[1])])
        return X_df
    
    def save(self, path='outputs/preprocessor.pkl'):
        joblib.dump(self.preprocessor, path)
    
    def load(self, path='outputs/preprocessor.pkl'):
        self.preprocessor = joblib.load(path)
        self.is_fitted = True

if __name__ == "__main__":
    df = pd.read_csv('data/merged_data.csv')
    processor = DataProcessor()
    X_proc, y = processor.fit_transform(df)
    print("X shape:", X_proc.shape)
    processor.save()