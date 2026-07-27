import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib


# Configuration de la page

st.set_page_config(page_title="S3d Ingénierie | Prédicteur HTL de Biomasse",
                   page_icon="🔬", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Theme global et variables */
    :root {
        --s3d-primary: #0083B0;
        --s3d-secondary: #00B4DB;
        --bg-card: #1E2530;
        --border-color: #2E3846;
        --text-color: #F0F2F5;
    }
    
    .stApp {
        background-color: #0E1117;
        color: var(--text-color);
    }

    /* Cartes de résultats */
    .metric-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .metric-card h4 {
        color: #A0AEC0;
        font-size: 0.9rem;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00B4DB;
    }

    /* Ajustement des conteneurs d'entrées */
    .stNumberInput input {
        background-color: #1A202C !important;
        color: #FFFFFF !important;
        border-radius: 6px;
    }
    
    /* Bouton principal */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0083B0 0%, #00B4DB 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 180, 219, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models_dict.pkl")
    return joblib.load(model_path)

models_dict = load_models()

# Tête de l'application

head_col1, head_col2 = st.columns([1, 4])

with head_col1:
    # Gestion sécurisée du logo s3d
    logo_path = "logo-S3D_3000x2000 (2).png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("### **S3d**\nIngénierie")

with head_col2:
    st.title("🔬 Prédicteur HTL")
    st.markdown("*Plateforme de prédiction du procédé de liquéfaction hydrothermale (HTL)*")

st.markdown("---")

# Organisation de l'interface en colonnes d'entrées

col_inputs, col_outputs = st.columns([1.1, 1])

with col_inputs:
    st.header("📋 Paramètres d'Entrée")
    
    # Onglets pour une organisation plus claire
    tab1, tab2, tab3 = st.tabs(["⚙️ Procédé & Solvant", "🧪 Analyse Ultime", "🧬 Analyse Biochimique"])
    
    with tab1:
        st.subheader("Conditions de Réaction")
        c1, c2 = st.columns(2)
        with c1:
            temp = st.number_input("Température [K]", value=573.15)
            time = st.number_input("Temps de séjour [min]", value=30.0)
        with c2:
            biomass_pct = st.number_input("Biomasse %", value=10.0)
            cat_qty = st.number_input("Quantité_Catalyseur", value=0.0)
            
        st.subheader("Solvant (% masse)")
        s1, s2, s3 = st.columns(3)
        with s1:
            water = st.number_input("Water_%", value=100.0, min_value=0.0, max_value=100.0)
        with s2:
            ethanol = st.number_input("Ethanol_%", value=0.0, min_value=0.0, max_value=100.0)
        with s3:
            methanol = st.number_input("Methanol_%", value=0.0, min_value=0.0, max_value=100.0)
            
        # Avertissement si le total solvant n'est pas à 100%
        solvent_total = water + ethanol + methanol
        if abs(solvent_total - 100.0) > 0.01:
            st.warning(f"⚠️ Le total du mélange solvant est de **{solvent_total:.1f}%** (attendu: 100%)")

    with tab2:
        st.subheader("Analyse Ultime & Propriétés")
        u1, u2 = st.columns(2)
        with u1:
            hc_feed = st.number_input("H/C_feed", value=1.5)
            oc_feed = st.number_input("O/C_feed", value=0.6)
            n = st.number_input("N", value=0.5)
            s = st.number_input("S", value=0.1)
        with u2:
            ash = st.number_input("Cendres [wt%]", value=2.0)
            pcs = st.number_input("PCS [MJ/kg]", value=19.5)

    with tab3:
        st.subheader("Composition Biochimique")
        b1, b2 = st.columns(2)
        with b1:
            protein = st.number_input("Protéines", value=10.0, step=1.0)
            lipid = st.number_input("Lipides", value=5.0, step=1.0)
            carbs = st.number_input("Glucides", value=50.0, step=1.0)
        with b2:
            lignin = st.number_input("Lignine", value=15.0, step=1.0)
            hemicellulose = st.number_input("Hemicellulose", value=10.0, step=1.0)
            cellulose = st.number_input("Cellulose", value=10.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🚀 CALCULER LES PRÉDICTIONS", type="primary")


# Bouton de prédiction et Outputs
with col_outputs:
    st.header("📊 Prédictions & Analyses")
    
    if btn_predict and models_loaded:
        # Construction exacte du DataFrame attendu par le modèle
        input_data = pd.DataFrame([[
            temp, time, biomass_pct, hc_feed, n, s, oc_feed, ash, pcs,
            water, ethanol, methanol, cat_qty, protein, lipid, carbs,
            lignin, hemicellulose, cellulose
        ]], columns=[
            'Température [K]', 'Temps de séjour [min]', 'Biomasse %', 'H/C_feed', 'N', 'S', 
            'O/C_feed', 'Cendres [wt%]', 'PCS [MJ/kg]', 'Water_%', 'Ethanol_%', 'Methanol_%', 
            'Quantité_Catalyseur', 'Protéines', 'Lipides', 'Glucides', 'Lignine', 'Hemicellulose', 'Cellulose'
        ])

        st.success("✅ Calcul effectué avec succès !")
        
        # Structure d'affichage dynamic sous forme de grille de cartes
        predictions = {}
        for target, model in models_dict.items():
            pred = model.predict(input_data)[0]
            predictions[target] = float(pred)

        # Affichage des métriques en grille 2 ou 3 colonnes selon le nombre de cibles
        pred_cols = st.columns(2)
        for idx, (target, val) in enumerate(predictions.items()):
            col_target = pred_cols[idx % 2]
            with col_target:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{target}</h4>
                    <div class="value">{val:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

        # Visualisation interactive Plotly des prédictions (ex: Diagramme à barres si plusieurs sorties)
        if len(predictions) > 1:
            st.subheader("Comparatif des Rendements / Propriétés")
            df_preds = pd.DataFrame(list(predictions.items()), columns=['Cible', 'Valeur'])
            
            fig = px.bar(
                df_preds, 
                x='Cible', 
                y='Valeur', 
                color='Cible',
                text_auto='.2f',
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Ajustez vos paramètres à gauche puis cliquez sur **Calculer les Prédictions**.")
        
        # Affichage de l'image explicative des produits si disponible
        products_img_path = "images/Products.png"
        if os.path.exists(products_img_path):
            st.image(products_img_path, caption="Schéma des produits HTL", use_container_width=True)


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #718096; font-size: 0.85rem;'>"
    "© s3d ingénierie — Outil d'optimisation Machine Learning HTL"
    "</div>", 
    unsafe_allow_html=True
)