import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Biomass HTL Predictor", layout="wide")

st.title("🔬 Biomass Hydrothermal Liquefaction Conversion Predictor")
st.write("Entrez les conditions opératoires et la composition biochimique pour prédire les rendements et propriétés.")

# Chargement du dictionnaire de modèles
@st.cache_resource
def load_models():
    return joblib.load("models_dict.pkl")

models_dict = load_models()

# Organisation de l'interface en 2 colonnes d'entrées
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Conditions de Réaction")
    temp = st.number_input("Température [K]", value=573.15)
    time = st.number_input("Temps de séjour [min]", value=30.0)
    biomass_pct = st.number_input("Biomasse %", value=10.0)
    water = st.number_input("Water_%", value=100.0)
    ethanol = st.number_input("Ethanol_%", value=0.0)
    methanol = st.number_input("Methanol_%", value=0.0)
    cat_qty = st.number_input("Quantité_Catalyseur", value=0.0)

with col2:
    st.subheader("🧪 Propriétés de la Biomasse")
    hc_feed = st.number_input("H/C_feed", value=1.5)
    oc_feed = st.number_input("O/C_feed", value=0.6)
    n = st.number_input("N", value=0.5)
    s = st.number_input("S", value=0.1)
    ash = st.number_input("Cendres [wt%]", value=2.0)
    pcs = st.number_input("PCS [MJ/kg]", value=19.5)
    protein = st.number_input("Protéines", value=10.0)
    lipid = st.number_input("Lipides", value=5.0)
    carbs = st.number_input("Glucides", value=50.0)
    lignin = st.number_input("Lignine", value=15.0)
    hemicellulose = st.number_input("Hemicellulose", value=10.0)
    cellulose = st.number_input("Cellulose", value=10.0)

# Bouton de prédiction
if st.button("🚀 Calculer les Prédictions", type="primary"):
    # Construction du DataFrame avec les noms exacts des colonnes
    input_data = pd.DataFrame([[
        temp, time, biomass_pct, hc_feed, n, s, oc_feed, ash, pcs,
        water, ethanol, methanol, cat_qty, protein, lipid, carbs,
        lignin, hemicellulose, cellulose
    ]], columns=[
        'Température [K]', 'Temps de séjour [min]', 'Biomasse %', 'H/C_feed', 'N', 'S', 
        'O/C_feed', 'Cendres [wt%]', 'PCS [MJ/kg]', 'Water_%', 'Ethanol_%', 'Methanol_%', 
        'Quantité_Catalyseur', 'Protéines', 'Lipides', 'Glucides', 'Lignine', 'Hemicellulose', 'Cellulose'
    ])
    
    st.markdown("---")
    st.subheader("📊 Résultats Prédits")
    
    # Affichage des résultats sous forme de métriques
    res_cols = st.columns(3)
    idx = 0
    
    for target, model in models_dict.items():
        pred = model.predict(input_data)[0]
        col_target = res_cols[idx % 3]
        col_target.metric(label=target, value=f"{round(float(pred), 3)}")
        idx += 1