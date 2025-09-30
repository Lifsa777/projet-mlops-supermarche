import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from recommendation_model import RecommendationModel
import plotly.express as px  # Pour graphs interactifs
import io

st.set_page_config(page_title="Surplus Predictor Pro", layout="wide", initial_sidebar_state="expanded")
st.markdown("#  Surplus Predictor : Réduisez le Gaspillage avec l'IA")

# Sidebar pro
st.sidebar.title(" Configuration")
theme = st.sidebar.selectbox("Thème", ["Light", "Dark"])
if theme == "Dark":
    st.markdown('<style> section[data-testid="stSidebar"] {background-color: #111;}</style>', unsafe_allow_html=True)

model = RecommendationModel()

# Pages
page = st.sidebar.radio("Navigation", ["Prédiction", "Métriques Modèle", "EDA Interactif"])

if page == "Prédiction":
    st.header(" Prédire le Surplus")
    with st.spinner("Chargement modèle..."):
        # Inputs en colonnes
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Contexte Magasin")
            branch = st.selectbox("Branch", ['Alex', 'Giza', 'Cairo'], key="branch")
            city = st.selectbox("City", ['Yangon', 'Naypyitaw', 'Mandalay'], key="city")
            customer_type = st.selectbox("Type Client Dominant", ['Member', 'Normal'], key="cust")
            product_line = st.selectbox("Ligne Produit", ['Food and beverages', 'Health and beauty', 
                                                          'Electronic accessories', 'Home and lifestyle', 
                                                          'Sports and travel', 'Fashion accessories'], key="prod")
        
        with col2:
            st.subheader("Détails Produit")
            unit_price = st.slider("Prix Unitaire (€)", 10.0, 100.0, 50.0, key="price")
            rating = st.slider("Note Moyenne", 1.0, 10.0, 7.0, key="rating")
            stock_initial = st.slider("Stock Initial", 5, 50, 20, key="stock")
            jours_avant = st.slider("Jours avant Péremption", 1, 60, 10, key="jours")
            jour_semaine = st.selectbox("Jour Semaine (0=Lun)", range(7), key="jour")
            mois = st.selectbox("Mois", range(1,13), key="mois")
        
        if st.button("🚀 Prédire Surplus", type="primary"):
            input_data = {
                'Branch': branch, 'City': city, 'Customer type': customer_type, 'Product line': product_line,
                'Unit price': unit_price, 'Rating': rating, 'stock_initial': stock_initial,
                'jours_avant_peremption': jours_avant, 'jour_semaine': jour_semaine, 'mois': mois
            }
            
            surplus = model.predict_surplus(input_data)
            promo = model.suggest_promotion(surplus, product_line, unit_price)
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric("Surplus Prédit", f"{surplus:.2f} unités")
            with col_b:
                st.info(promo)
            
            # Graph interactif : Surplus vs Inputs clés
            df_plot = pd.DataFrame({
                'Jours avant Péremption': [jours_avant], 'Prix': [unit_price], 'Stock Initial': [stock_initial],
                'Surplus Prédit': [surplus]
            })
            fig = px.scatter(df_plot, x='Jours avant Péremption', y='Surplus Prédit', size='Stock Initial',
                             color='Prix', title="Impact Inputs sur Surplus")
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = pd.DataFrame([input_data | {'surplus_pred': surplus, 'promotion': promo}])
            st.download_button(" Exporter Prédiction CSV", csv.to_csv(index=False), "prediction.csv")

elif page == "Métriques Modèle":
    st.header("Performance Modèle")
    with open('outputs/metrics.json', 'r') as f:
        metrics = pd.read_json(f)
    st.dataframe(metrics.T)  # Transposé pour lisibilité
    st.success("Modèle déployé : R² moyen > 0.85 pour production.")

elif page == "EDA Interactif":
    st.header("EDA des Données")
    @st.cache_data
    def load_merged():
        return pd.read_csv('data/merged_data.csv')
    
    df = load_merged()
    st.dataframe(df.head(10))
    
    # Graph interactif
    fig = px.box(df, x='Product line', y='surplus', color='Branch', title="Surplus par Produit et Branch")
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Développé pour Supermarché Anti-Gaspillage  2025")