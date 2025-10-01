import streamlit as st
import pandas as pd
import joblib
import os

# Configuration de la page
st.set_page_config(page_title="Prédiction de Surplus", layout="centered", page_icon="🛒")

# --- Styles (Uniformisation Bleu & Corrections HTML/CSS) ---
st.markdown(
    """
    <style>
    :root{
        --bg: #e6faff;              /* bleu clair */
        --card: #ffffff;            /* carte blanche pour fort contraste */
        --accent-blue: #0369a1;     /* BLEU VISIBLE (Couleur principale) */
        --text: #021324;            /* texte sombre pour bon contraste */
        --muted: #274153;
        --radius: 10px;
    }
    html, body, [class*="css"] {
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .stApp { background: linear-gradient(180deg, var(--bg) 0%, #f0fbff 100%) !important; }
    .block-container { max-width: 920px; padding: 20px 28px; background: transparent; color: var(--text) !important; }

    .app-header {
        /* Uniformisation : teintes de bleu */
        background: linear-gradient(90deg, rgba(3,105,161,0.06), rgba(2,19,36,0.03));
        border-radius: var(--radius);
        padding: 14px 18px;
        margin-bottom: 18px;
        color: var(--text);
    }
    .app-title { font-size: 22px; font-weight:700; margin:0; color: var(--text) !important; }
    .app-sub { color: var(--muted); margin-top:6px; font-size:13px; }

    .stForm, form {
        background: var(--card);
        padding: 16px;
        border-radius: var(--radius);
        box-shadow: 0 6px 18px rgba(7,16,41,0.06);
    }

    /* Bouton principal : Passage au BLEU */
    button, .stButton>button, div.stButton>button {
        background: var(--accent-blue) !important; 
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        font-weight:600 !important;
        box-shadow: none !important;
    }
    button:hover, .stButton>button:hover { background: #014c77 !important; } /* Bleu foncé au survol */

    /* Forcer couleur sombre pour textes/labels/entêtes */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText, .block-container, .app-sub {
        color: var(--text) !important;
    }

    /* Inputs/labels internes Streamlit */
    label[for], .css-1v3fvcr, .css-1kyxreq, .css-1d391kg, .css-1n76uvr, .css-1nn9j9p {
        color: var(--text) !important;
    }

    /* Metrics: forcer couleur foncée et taille */
    .stMetric, .stMetric * { color: var(--text) !important; }
    .stMetricValue, .stMetricValue > div, .stMetricValue span {
        color: var(--text) !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    .stMetricLabel, .stMetricDelta { color: var(--text) !important; font-weight: 700 !important; }

    .small-muted { color: var(--muted) !important; font-size:13px; }

    /* Carte résultat HTML personnalisée */
    .result-card {
        background: var(--card);
        color: var(--text);
        padding: 14px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(7,16,41,0.06);
        margin-top: 12px;
        margin-bottom: 12px;
    }
    .result-title { font-size: 18px; font-weight:700; color: var(--text); margin-bottom: 8px; }
    .result-value { font-size: 34px; font-weight:800; color: var(--text); margin-bottom: 6px; }
    .result-meta { color: var(--muted); font-size:13px; }
    
    /* Couleurs des Recommandations */
    .recommendation { margin-top:10px; padding:10px; border-radius:8px; font-weight:600; }
    .recommendation.crit { background:#ffebeb; color:#9b1c1c; } /* Rouge pour Critique */
    .recommendation.high { background:#fffbeb; color:#92400e; } /* Jaune/Orange pour Élevé */
    .recommendation.mod { background:#f0f9ff; color:#0369a1; } /* Bleu très clair pour Modéré */
    .recommendation.low { background:#ecfdf5; color:#065f46; } /* Vert clair pour Faible risque */

    /* Force les couleurs du graphique à être visibles même avec le thème sombre */
    .css-1ht1c8r, .css-1fay8r6, .css-1n04f3n { color: var(--text) !important; } 

    </style>
    """ ,
    unsafe_allow_html=True,
)

# --- En-tête personnalisé ---
st.markdown(
    """
    <div class="app-header">
      <h1 class="app-title">Prédiction de Surplus en Supermarché</h1>
      <div class="app-sub">Estimez le surplus produit et recevez des recommandations d'action.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Fonctions de Chargement ---
@st.cache_resource
def load_model():
    """Charge le modèle de prédiction une seule fois."""
    model_path = os.path.join('saved_model', 'surplus_predict_model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Fichier modèle '{model_path}' non trouvé. Exécutez 'train.py' puis relancez.")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

@st.cache_data
def get_unique_values_from_data():
    """Charge les données pour obtenir les listes de valeurs uniques pour les sélecteurs."""
    data_path = os.path.join('data', 'donnees_traitees.csv')
    if not os.path.exists(data_path):
        st.error(f"Fichier '{data_path}' non trouvé. Exécutez 'train.py' pour générer les données prétraitées.")
        return None
    try:
        df = pd.read_csv(data_path)
        return {
            'branch': sorted(df['Branch'].dropna().unique().tolist()),
            'city': sorted(df['City'].dropna().unique().tolist()),
            'product_line': sorted(df['Product line'].dropna().unique().tolist())
        }
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

# --- Chargement ---
model = load_model()
unique_values = get_unique_values_from_data()

# --- Interface Utilisateur ---
st.title("Prédiction de Surplus en Supermarché")

if model and unique_values:
    with st.form("prediction_form"):
        st.header("Entrez les informations du produit")
        st.markdown('<div class="small-muted">Remplissez les champs ci‑dessous puis lancez la prédiction.</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            product_line = st.selectbox("Ligne de Produit", options=unique_values['product_line'])
            branch = st.selectbox("Branche", options=unique_values['branch'])
            city = st.selectbox("Ville", options=unique_values['city'])
            jour_semaine = st.selectbox(
                "Jour de la semaine",
                options=list(range(7)),
                format_func=lambda x: ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'][x]
            )

        with col2:
            stock_initial = st.number_input("Stock Initial", min_value=1, max_value=10000, value=25, step=1)
            unit_price = st.slider("Prix Unitaire (€)", min_value=0.0, max_value=100000.0, value=55.0, step=0.1)
            jours_avant_peremption = st.slider("Jours avant Péremption", min_value=1, max_value=365, value=15)
            mois = st.select_slider("Mois", options=list(range(1, 13)), value=pd.Timestamp.now().month)

        submit_button = st.form_submit_button(label="Lancer la Prédiction", use_container_width=True)

    if submit_button:
        # Préparation des données pour la prédiction
        input_data = {
            'Branch': [branch],
            'City': [city],
            'Product line': [product_line],
            'Unit price': [unit_price],
            'stock_initial': [stock_initial],
            'jours_avant_peremption': [jours_avant_peremption],
            'jour_semaine': [jour_semaine],
            'mois': [mois]
        }
        input_df = pd.DataFrame(input_data)

        with st.spinner('Analyse en cours...'):
            try:
                prediction = model.predict(input_df)
                predicted_surplus = int(round(float(prediction[0])))
                if predicted_surplus < 0:
                    predicted_surplus = 0
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                predicted_surplus = None

        if predicted_surplus is not None:
            # 1. Calcul des indicateurs clés
            ratio = (predicted_surplus / stock_initial) if stock_initial > 0 else 0
            value_at_risk = predicted_surplus * unit_price
            
            # NOUVEAU: Définition de l'urgence de péremption (5 jours ou moins)
            is_peremption_imminente = jours_avant_peremption <= 5
            
            # Affichage clair du résultat (avec correction du formatage)
            html = f"""
            <div class="result-card">
              <div class="result-title">Résultat de la Prédiction</div>
              <div class="result-value">{predicted_surplus} unités</div>
              <div class="result-meta">
                  Stock initial: {stock_initial} unités • Prix unitaire: €{unit_price:,.2f}<br>
                  <b>Valeur du Risque (perte potentielle): €{value_at_risk:,.2f}</b> • Jours avant péremption: {jours_avant_peremption} jours
              </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

            # 2. Logique de Recommandation OPTIMISÉE (Ratio + Valeur Monétaire + Urgence)
            rec_msg = ""
            rec_class = "low"
            
            # Correction des doubles astérisques avec <b>
            urgence_suffix = f" <b>(Urgence : {jours_avant_peremption} jours restants!)</b>" if is_peremption_imminente else ""

            # Critères combinés
            if ratio >= 0.5 or value_at_risk >= 500 or (is_peremption_imminente and ratio >= 0.3): 
                rec_msg = f"<b>Critique</b> — Risque maximal. Action immédiate : <b>Liquidation rapide (-40% à -60%)</b> ou redistribution urgente.{urgence_suffix}"
                rec_class = "crit"
            elif ratio >= 0.3 or value_at_risk >= 100 or (is_peremption_imminente and ratio >= 0.15): 
                rec_msg = f"<b>Élevé</b> — Surplus important. Action conseillée : <b>Promotion ciblée (-20% à -30%)</b> et mise en avant.{urgence_suffix}"
                rec_class = "high"
            elif ratio >= 0.1: 
                rec_msg = "<b>Modéré</b> — Surplus détecté. Action suggérée : <b>Petite promotion (-10%)</b> ou vérification du facing et des stocks."
                rec_class = "mod"
            else:
                rec_msg = "<b>Faible</b> — Surplus gérable. Aucun ajustement immédiat nécessaire. Surveillance du stock."
                rec_class = "low"

            st.markdown(f"<div class='recommendation {rec_class}'>{rec_msg}</div>", unsafe_allow_html=True)

else:
    st.error("L'application n'a pas pu démarrer. Vérifiez les fichiers 'data/' et 'saved_model/'.")