import pandas as pd
import joblib
from data_processor import DataProcessor

class RecommendationModel:
    def __init__(self, model_path='outputs/best_model.pkl', preprocessor_path='outputs/preprocessor.pkl'):
        self.model = joblib.load(model_path)
        self.processor = DataProcessor()
        self.processor.load(preprocessor_path)
    
    def predict_surplus(self, input_data):
        df_input = pd.DataFrame([input_data])
        X_input = self.processor.transform(df_input)
        return self.model.predict(X_input)[0]
    
    def suggest_promotion(self, surplus_pred, product_line, unit_price):
        seuil = unit_price * 0.1  # Dynamique : 10% du prix comme seuil bas
        if surplus_pred > seuil * 2:
            promo = f"🚨 Promotion urgente : -30% sur {product_line} (prix: {unit_price:.2f}€, risque haut)"
        elif surplus_pred > seuil:
            promo = f"⚠️ Promotion modérée : -15% sur {product_line} (prix: {unit_price:.2f}€)"
        else:
            promo = f"✅ Pas de promo nécessaire pour {product_line} (surplus bas)."
        return promo

if __name__ == "__main__":
    model = RecommendationModel()
    input_ex = {'Branch': 'Alex', 'City': 'Yangon', 'Customer type': 'Member', 'Product line': 'Food and beverages',
                'Unit price': 50, 'Rating': 8.0, 'stock_initial': 20, 'jours_avant_peremption': 5,
                'jour_semaine': 3, 'mois': 6}
    surplus = model.predict_surplus(input_ex)
    promo = model.suggest_promotion(surplus, input_ex['Product line'], input_ex['Unit price'])
    print(f"Surplus: {surplus:.2f} | {promo}")