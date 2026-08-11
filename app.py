mport os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="S3d Ingénierie | Prédicteur HTL de Biomasse",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    /* Theme global et variables S3D */
    :root {
        --s3d-green: #059669;
        --s3d-green-hover: #047857;
        --s3d-green-light: #ECFDF5;
        --bg-app: #F8FAFC;
        --bg-card: #FFFFFF;
        --border-color: #E2E8F0;
        --text-dark: #0F172A;
        --text-muted: #475569;
    }
    
    .stApp {
        background-color: var(--bg-app);
        color: var(--text-dark);
    }

    /* Cartes de résultats */
    .metric-card {
        background-color: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.12);
        border-color: var(--s3d-green);
    }
    .metric-card h4 {
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--s3d-green);
    }

    /* Correctif des champs d'entrée */
    .stNumberInput input {
        background-color: #FFFFFF !important;
        color: var(--text-dark) !important;
        border-radius: 8px;
        border: 1px solid #CBD5E1 !important;
    }
    
    /* Personnalisation des Onglets */
    .stTabs [aria-selected="true"] {
        color: var(--s3d-green) !important;
        border-bottom-color: var(--s3d-green) !important;
        font-weight: bold;
    }
    
    /* Bouton principal */
    div.stButton > button:first-child {
        background-color: var(--s3d-green) !important;
        color: white !important;
        font-weight: bold;
        font-size: 1.05rem;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 6px rgba(5, 150, 105, 0.2);
    }
    div.stButton > button:first-child:hover {
        background-color: var(--s3d-green-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(5, 150, 105, 0.35);
    }
</style>
""",
    unsafe_allow_html=True,
)


# Chargement du modèle
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models_dict.pkl")
    return joblib.load(model_path)


models_dict = load_models()
models_loaded = models_dict is not None

# Tête de l'application
head_col1, head_col2 = st.columns([1, 4])

with head_col1:
    logo_path = "logo-S3D_3000x2000 (2).png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.markdown("### **S3d**\nIngénierie")

with head_col2:
    st.title("🔬 Prédicteur HTL")
    st.markdown(
        "*Plateforme de prédiction du procédé de liquéfaction hydrothermale"
        " (HTL)*"
    )

st.markdown("---")

# Organisation de l'interface en colonnes d'entrées
col_inputs, col_outputs = st.columns([1.1, 1])

with col_inputs:
    st.header("📋 Paramètres d'Entrée")

    tab1, tab2, tab3 = st.tabs(
        [
            "⚙️ Procédé & Solvant",
            "🧪 Analyse Ultime",
            "🧬 Analyse Biochimique",
        ]
    )

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
            water = st.number_input(
                "Water_%", value=100.0, min_value=0.0, max_value=100.0
            )
        with s2:
            ethanol = st.number_input(
                "Ethanol_%", value=0.0, min_value=0.0, max_value=100.0
            )
        with s3:
            methanol = st.number_input(
                "Methanol_%", value=0.0, min_value=0.0, max_value=100.0
            )

        solvent_total = water + ethanol + methanol
        if abs(solvent_total - 100.0) > 0.01:
            st.warning(
                f"⚠️ Le total du mélange solvant est de **{solvent_total:.1f}%**"
                " (attendu: 100%)"
            )

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
            hemicellulose = st.number_input(
                "Hemicellulose", value=10.0, step=1.0
            )
            cellulose = st.number_input("Cellulose", value=10.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    btn_predict = st.button("🚀 CALCULER LES PRÉDICTIONS", type="primary")


# Bouton de prédiction et Outputs
with col_outputs:
    st.header("📊 Prédictions & Analyses")

    if btn_predict and models_loaded:
        input_data = pd.DataFrame(
            [
                [
                    temp,
                    time,
                    biomass_pct,
                    hc_feed,
                    n,
                    s,
                    oc_feed,
                    ash,
                    pcs,
                    water,
                    ethanol,
                    methanol,
                    cat_qty,
                    protein,
                    lipid,
                    carbs,
                    lignin,
                    hemicellulose,
                    cellulose,
                ]
            ],
            columns=[
                "Température (K)",
                "Temps de séjour (min)",
                "Biomasse %",
                "H/C_feed",
                "N",
                "S",
                "O/C_feed",
                "Cendres (wt%)",
                "PCS (MJ/kg)",
                "Water_%",
                "Ethanol_%",
                "Methanol_%",
                "Quantité_Catalyseur",
                "Protéines",
                "Lipides",
                "Glucides",
                "Lignine",
                "Hemicellulose",
                "Cellulose",
            ],
        )

        st.success("✅ Calcul effectué avec succès !")

        predictions = {}
        for target, model in models_dict.items():
            pred = model.predict(input_data)[0]
            predictions[target] = float(pred)

        pred_cols = st.columns(2)
        for idx, (target, val) in enumerate(predictions.items()):
            col_target = pred_cols[idx % 2]
            with col_target:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <h4>{target}</h4>
                    <div class="value">{val:.3f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        rendements_dict = {}
        properties_dict = {}

        # Patrón para detectar PCS, HHV o elementos químicos C, H, N, S, O aislados
        prop_pattern = (
            r"\b(pcs|c|h|n|s|o|carbon|hydrogen|nitrogen|oxygen|sulfur)\b"
        )

        for target, val in predictions.items():
            t_lower = target.lower()

            # Rendements
            if "rendement" in t_lower or "yield" in t_lower:
                rendements_dict[target] = val
            # Propriétés
            elif (
                "pcs" in t_lower
                or "hhv" in t_lower
                or re.search(prop_pattern, t_lower)
            ):
                properties_dict[target] = val
            # Por defecto
            else:
                rendements_dict[target] = val

        # -------------------------------------------------------------
        # DESPLIEGUE EN 2 ONGLETS / GRÁFICOS SEPARADOS
        # -------------------------------------------------------------
        tab_plot1, tab_plot2 = st.tabs(
            ["📈 Rendements (%)", "🧪 Biocrude & PCS (C, H, N, S, O, PCS)"]
        )

        # PLOT 1: RENDEMENTS
        with tab_plot1:
            if rendements_dict:
                df_rend = pd.DataFrame(
                    list(rendements_dict.items()),
                    columns=["Produit", "Rendement (%)"],
                )
                fig_rend = px.bar(
                    df_rend,
                    x="Produit",
                    y="Rendement (%)",
                    color="Produit",
                    text_auto=".2f",
                    color_discrete_sequence=[
                        "#10B981",
                        "#0284C7",
                        "#64748B",
                        "#B45309",
                    ],
                    title="Rendement des phases HTL (%)",
                )
                fig_rend.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_rend, use_container_width=True)
            else:
                st.info("Aucun rendement détecté.")

        # PLOT 2: PROPRIÉTÉS (BIOCRUDE C, H, N, S, O + PCS)
        with tab_plot2:
            if properties_dict:
                df_prop = pd.DataFrame(
                    list(properties_dict.items()),
                    columns=["Propriété", "Valeur"],
                )
                fig_prop = px.bar(
                    df_prop,
                    x="Propriété",
                    y="Valeur",
                    color="Propriété",
                    text_auto=".2f",
                    color_discrete_sequence=[
                        "#059669",
                        "#0D9488",
                        "#3B82F6",
                        "#8B5CF6",
                        "#D97706",
                        "#EC4899",
                    ],
                    title="Composés du Biocrude (C, H, N, S, O) et PCS",
                )
                fig_prop.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_prop, use_container_width=True)
            else:
                st.info("Aucune propriété détectée.")

    else:
        st.info(
            "👈 Ajustez vos paramètres à gauche puis cliquez sur **CALCULER LES"
            " PRÉDICTIONS**."
        )

        products_img_path = "images/Products.png"
        if os.path.exists(products_img_path):
            st.image(
                products_img_path,
                caption="Schéma des produits HTL",
                use_container_width=True,
            )

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "© s3d ingénierie — Outil d'optimisation Machine Learning HTL"
    "</div>",
    unsafe_allow_html=True,
)
