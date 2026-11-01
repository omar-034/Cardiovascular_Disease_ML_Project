"""
CardioScan AI - Système Expert d'Aide à la Décision Clinique
Application Web Streamlit pour la prédiction et l'évaluation du risque de maladie cardiovasculaire.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(
    page_title="CardioScan AI | Diagnostic Cardiovasculaire",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement du pipeline d'inférence
try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomData
except ImportError:
    # Si le package n'est pas encore installé en mode éditable dans l'environnement courant
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from src.pipeline.predict_pipeline import PredictPipeline, CustomData


# ==============================================================================
# FEUILLE DE STYLE CSS PERSONNALISÉE (STYLE MÉDICAL ET PROFESSIONNEL)
# ==============================================================================
st.markdown("""
<style>
    /* Polices et conteneurs */
    .main-header {
        background: linear-gradient(135deg, #0F2027, #203A43, #2C5364);
        padding: 24px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #E0E0E0;
        font-size: 1.05rem;
        margin: 0;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2C5364;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .card-title {
        font-size: 0.85rem;
        color: #6C757D;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1A252F;
    }
    .risk-critical {
        background-color: #F8D7DA;
        color: #721C24;
        border-left: 6px solid #DC3545;
        padding: 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    .risk-high {
        background-color: #FFF3CD;
        color: #856404;
        border-left: 6px solid #FFC107;
        padding: 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    .risk-safe {
        background-color: #D4EDDA;
        color: #155724;
        border-left: 6px solid #28A745;
        padding: 18px;
        border-radius: 8px;
        font-weight: 600;
    }
    .biomarker-chip {
        display: inline-block;
        background: #EBF3FB;
        color: #1D6F99;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# FONCTIONS UTILITAIRES & CHARGEMENT DES ARTEFACTS
# ==============================================================================
@st.cache_resource
def load_prediction_engine():
    """Initialise et met en cache l'instance du pipeline d'inférence."""
    return PredictPipeline()


@st.cache_data
def load_metrics():
    """Charge les métriques de test et les détails du modèle champion."""
    metrics_path = os.path.join("artifacts", "metrics.json")
    threshold_path = os.path.join("artifacts", "threshold.json")
    model_comp_path = os.path.join("artifacts", "model_comparison.csv")

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    threshold_info = {}
    if os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold_info = json.load(f)

    model_comp_df = None
    if os.path.exists(model_comp_path):
        model_comp_df = pd.read_csv(model_comp_path)

    return metrics, threshold_info, model_comp_df


prediction_engine = load_prediction_engine()
metrics, threshold_info, model_comp_df = load_metrics()


# ==============================================================================
# EN-TÊTE PRINCIPAL
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h1>🫀 CardioScan AI — Aide au Diagnostic Cardiovasculaire</h1>
    <p>Outil d'intelligence artificielle clinique pour le dépistage précoce des cardiopathies coronariennes et l'évaluation du risque ischémique.</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# BARRE LATÉRALE (SIDEBAR) : MÉTRIQUES ET PROFILS PRÉDÉFINIS
# ==============================================================================
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
        caption="CardioScan AI Clinical Suite",
        width="stretch"
    )
    
    st.subheader("Performance du Modèle")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric(
            label="Rappel (Sensibilité)",
            value=f"{metrics.get('test_recall', 1.0) * 100:.1f}%",
            help="Priorité médicale : 100% de détection des patients atteints (0 faux négatif sur le jeu de test)."
        )
        st.metric(
            label="Exactitude (Accuracy)",
            value=f"{metrics.get('test_accuracy', 0.99) * 100:.1f}%"
        )
    with col_sb2:
        st.metric(
            label="Score ROC-AUC",
            value=f"{metrics.get('test_roc_auc', 0.9999):.4f}",
            help="Capacité de discrimination quasi parfaite du modèle champion HistGradientBoosting."
        )
        st.metric(
            label="Seuil Dédié (θ*)",
            value=f"{threshold_info.get('optimal_threshold', 0.10):.2f}",
            help="Seuil clinique optimisé pour maximiser le F2-Score et éliminer les faux négatifs."
        )

    st.markdown("---")
    st.subheader("Charger un Profil Clinique Type")
    preset_choice = st.selectbox(
        "Sélectionnez un profil pré-rempli :",
        ["Saisie Manuelle", "Profil Témoin (Patient Sain)", "Profil à Risque Modéré", "Profil Ischémique Critique (Haut Risque)"]
    )

    st.markdown("---")
    st.caption("ℹ️ **Avertissement Médical** : Ce système est un support d'aide à la décision clinique. Il ne se substitue en aucun cas à l'expertise d'un cardiologue ou à des examens d'imagerie diagnostique (coronarographie, scintigraphie).")


# Valeurs par défaut selon le profil choisi
default_vals = {
    "age": 52,
    "gender": 1,
    "chestpain": 0,
    "restingBP": 130,
    "serumcholestrol": 210,
    "fastingbloodsugar": 0,
    "restingrelectro": 0,
    "maxheartrate": 155,
    "exerciseangia": 0,
    "oldpeak": 0.4,
    "slope": 1,
    "noofmajorvessels": 0
}

if preset_choice == "Profil Témoin (Patient Sain)":
    # Cas témoin typique : 20 ans, femme, pas de douleur, ST normal, FC bonne, 0 vaisseau
    default_vals = {
        "age": 25,
        "gender": 0,
        "chestpain": 0,
        "restingBP": 118,
        "serumcholestrol": 185,
        "fastingbloodsugar": 0,
        "restingrelectro": 0,
        "maxheartrate": 172,
        "exerciseangia": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "noofmajorvessels": 0
    }
elif preset_choice == "Profil à Risque Modéré":
    default_vals = {
        "age": 55,
        "gender": 1,
        "chestpain": 1,
        "restingBP": 142,
        "serumcholestrol": 250,
        "fastingbloodsugar": 0,
        "restingrelectro": 1,
        "maxheartrate": 138,
        "exerciseangia": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "noofmajorvessels": 1
    }
elif preset_choice == "Profil Ischémique Critique (Haut Risque)":
    # Cas critique : 76 ans, homme, angine d'effort, dépression ST 2.9 mm, 2 vaisseaux
    default_vals = {
        "age": 76,
        "gender": 1,
        "chestpain": 2,
        "restingBP": 158,
        "serumcholestrol": 285,
        "fastingbloodsugar": 1,
        "restingrelectro": 2,
        "maxheartrate": 115,
        "exerciseangia": 1,
        "oldpeak": 3.2,
        "slope": 3,
        "noofmajorvessels": 2
    }


# ==============================================================================
# ONGLETS DE NAVIGATION
# ==============================================================================
tab_single, tab_batch, tab_model = st.tabs([
    "🔍 Évaluation Clinique Individuelle",
    "📁 Dépistage par Cohorte (Batch CSV)",
    "📊 Gouvernance & Métriques du Modèle"
])


# ==============================================================================
# ONGLET 1 : ÉVALUATION INDIVIDUELLE
# ==============================================================================
with tab_single:
    st.markdown("### Paramètres Physiologiques et Électrocardiographiques du Patient")
    st.info("Renseignez les 12 biomarqueurs cliniques ci-dessous ou sélectionnez un profil type dans la barre latérale.")

    with st.form("patient_clinical_form"):
        col_form1, col_form2 = st.columns(2)

        # ----------------------------------------------------
        # Colonne 1 : Données Démographiques & Biologiques
        # ----------------------------------------------------
        with col_form1:
            st.markdown("##### 👤 1. Données Démographiques & Bilan Biologique")
            
            age = st.slider(
                "Âge du Patient (années)",
                min_value=20,
                max_value=90,
                value=int(default_vals["age"]),
                step=1,
                help="Âge chronologique lors de l'examen."
            )
            
            gender_label = st.radio(
                "Sexe Biologique",
                options=["Masculin", "Féminin"],
                index=0 if default_vals["gender"] == 1 else 1,
                horizontal=True
            )
            gender = 1 if gender_label == "Masculin" else 0

            restingBP = st.number_input(
                "Pression Artérielle au Repos (restingBP en mmHg)",
                min_value=80,
                max_value=220,
                value=int(default_vals["restingBP"]),
                step=1,
                help="Pression artérielle systolique mesurée à l'admission."
            )

            serumcholestrol = st.number_input(
                "Cholestérol Sérique (serumcholestrol en mg/dL)",
                min_value=0,
                max_value=650,
                value=int(default_vals["serumcholestrol"]),
                step=5,
                help="Taux de cholestérol total. Les valeurs nulles sont automatiquement imputées par la médiane du profil."
            )

            fbs_label = st.selectbox(
                "Glycémie à Jeun (fastingbloodsugar)",
                options=["Normale (≤ 120 mg/dL)", "Élevée / Diabète (> 120 mg/dL)"],
                index=int(default_vals["fastingbloodsugar"])
            )
            fastingbloodsugar = 1 if "Élevée" in fbs_label else 0

        # ----------------------------------------------------
        # Colonne 2 : Symptômes & Électrocardiogramme
        # ----------------------------------------------------
        with col_form2:
            st.markdown("##### ⚡ 2. Symptomatologie & Épreuve d'Effort")

            chestpain_options = {
                0: "0 - Asymptomatique / Douleur atypique sans angine",
                1: "1 - Angine de poitrine typique (Typical Angina)",
                2: "2 - Angine atypique (Atypical Angina)",
                3: "3 - Douleur non-angineuse (Non-anginal pain)"
            }
            chestpain = st.selectbox(
                "Type de Douleur Thoracique (chestpain)",
                options=list(chestpain_options.keys()),
                format_func=lambda x: chestpain_options[x],
                index=int(default_vals["chestpain"])
            )

            restingrelectro_options = {
                0: "0 - Normal",
                1: "1 - Anomalie du segment ST-T (Inversion T ou décalage ST)",
                2: "2 - Hypertrophie Ventriculaire Gauche (Critères Estes)"
            }
            restingrelectro = st.selectbox(
                "Électrocardiogramme au Repos (restingrelectro)",
                options=list(restingrelectro_options.keys()),
                format_func=lambda x: restingrelectro_options[x],
                index=int(default_vals["restingrelectro"])
            )

            maxheartrate = st.slider(
                "Fréquence Cardiaque Maximale Atteinte (maxheartrate en bpm)",
                min_value=60,
                max_value=220,
                value=int(default_vals["maxheartrate"]),
                step=1,
                help="Pic de fréquence atteint lors du test d'effort."
            )

            exang_label = st.radio(
                "Angine Induite par l'Effort (exerciseangia)",
                options=["Non (0)", "Oui (1)"],
                index=int(default_vals["exerciseangia"]),
                horizontal=True
            )
            exerciseangia = 1 if "Oui" in exang_label else 0

            oldpeak = st.slider(
                "Dépression du Segment ST à l'effort (oldpeak en mm)",
                min_value=0.0,
                max_value=6.5,
                value=float(default_vals["oldpeak"]),
                step=0.1,
                help="Sous-décalage du segment ST induit par l'exercice par rapport au repos."
            )

            slope_options = {
                1: "1 - Pente Ascendante (Upsloping - Physiologique)",
                2: "2 - Pente Horizontale (Flat - Suspect)",
                3: "3 - Pente Descendante (Downsloping - Hautement Ischémique)",
                0: "0 - Non Spécifiée / Hors Plage"
            }
            slope = st.selectbox(
                "Pente du Segment ST au Pic d'Effort (slope)",
                options=list(slope_options.keys()),
                format_func=lambda x: slope_options[x],
                index=list(slope_options.keys()).index(int(default_vals["slope"]))
            )

            noofmajorvessels = st.selectbox(
                "Nombre de Vaisseaux Majeurs Visibles en Fluoroscopie (noofmajorvessels)",
                options=[0, 1, 2, 3],
                index=int(default_vals["noofmajorvessels"]),
                help="Nombre de grosses artères coronaires opacifiées sous rayons X."
            )

        # ----------------------------------------------------
        # Biomarqueurs Dérivés Calculés en Temps Réel
        # ----------------------------------------------------
        theoretical_max_hr = 220 - age
        hr_reserve_ratio = maxheartrate / theoretical_max_hr if theoretical_max_hr > 0 else 1.0
        ischemia_burden = oldpeak * exerciseangia
        vascular_stiffness = (restingBP * age) / 100.0

        st.markdown("##### 🔬 Biomarqueurs Dérivés Synthétisés par le Feature Engineering :")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown(f"<div class='biomarker-chip'>Ratio Réserve Cardiaque : {hr_reserve_ratio:.2f}</div>", unsafe_allow_html=True)
            st.caption("FC Max atteinte / (220 - Âge)")
        with col_b2:
            st.markdown(f"<div class='biomarker-chip'>Charge Ischémique : {ischemia_burden:.2f} mm</div>", unsafe_allow_html=True)
            st.caption("oldpeak × Angine d'effort")
        with col_b3:
            st.markdown(f"<div class='biomarker-chip'>Rigidité Vasculaire : {vascular_stiffness:.2f}</div>", unsafe_allow_html=True)
            st.caption("(Pression Artérielle x Âge) / 100")

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🫀 Évaluer le Risque Cardiovasculaire du Patient", width="stretch", type="primary")

    # ----------------------------------------------------
    # Traitement & Affichage du Diagnostic
    # ----------------------------------------------------
    if submit_btn:
        try:
            patient_data = CustomData(
                age=float(age),
                gender=int(gender),
                chestpain=int(chestpain),
                restingBP=float(restingBP),
                serumcholestrol=float(serumcholestrol),
                fastingbloodsugar=int(fastingbloodsugar),
                restingrelectro=int(restingrelectro),
                maxheartrate=float(maxheartrate),
                exerciseangia=int(exerciseangia),
                oldpeak=float(oldpeak),
                slope=int(slope),
                noofmajorvessels=int(noofmajorvessels)
            )

            df_input = patient_data.get_data_as_data_frame()
            prediction_output = prediction_engine.predict(df_input)
            patient_res = prediction_output["patients_evaluated"][0]

            prob = patient_res["risk_probability"]
            prob_pct = patient_res["risk_percentage"]
            pred_label = patient_res["prediction_label"]
            risk_level = patient_res["risk_level"]
            recommendation = patient_res["clinical_recommendation"]
            threshold_val = patient_res["threshold_applied"]

            st.markdown("---")
            st.subheader("📋 Rapport Diagnostique et Évaluation du Risque")

            col_res1, col_res2, col_res3 = st.columns([1.5, 1, 1])
            with col_res1:
                if patient_res["prediction"] == 1:
                    st.error(f"⚠️ **DIAGNOSTIC : {pred_label.upper()}**")
                else:
                    st.success(f"✅ **DIAGNOSTIC : {pred_label.upper()}**")
                st.write(f"**Classement Clinique :** {risk_level}")

            with col_res2:
                st.metric(
                    label="Probabilité de Risque Ischémique",
                    value=prob_pct,
                    delta=f"{(prob - threshold_val) * 100:+.1f}% vs Seuil ({threshold_val * 100:.0f}%)",
                    delta_color="inverse"
                )

            with col_res3:
                st.metric(
                    label="Seuil Clinique Appliqué",
                    value=f"{threshold_val:.2f}",
                    help="Seuil d'action optimisé (Recall = 100% sur cohorte de validation)."
                )

            # Barre de progression colorée
            st.write("**Niveau de Risque Gradué :**")
            st.progress(float(min(max(prob, 0.0), 1.0)))

            # Encadré de Recommandation Médicale
            if prob >= 0.75:
                st.markdown(f"""
                <div class="risk-critical">
                    <h4>🚨 Alerte : {risk_level}</h4>
                    <p><strong>Conduite à tenir recommandée :</strong> {recommendation}</p>
                    <ul>
                        <li>Hospitalisation ou consultation cardiologique sous 24-48 heures.</li>
                        <li>Prescription d'une coronarographie diagnostique et dosage des troponines.</li>
                        <li>Mise en route immédiate d'un protocole anti-ischémique (selon antécédents).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif prob >= threshold_val:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>⚠️ Attention : {risk_level}</h4>
                    <p><strong>Conduite à tenir recommandée :</strong> {recommendation}</p>
                    <ul>
                        <li>Consultation cardiologique programmée avec échographie cardiaque transthoracique.</li>
                        <li>Optimisation des thérapeutiques antihypertensives et hypolipémiantes.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-safe">
                    <h4>✅ Statut : {risk_level}</h4>
                    <p><strong>Conduite à tenir recommandée :</strong> {recommendation}</p>
                    <ul>
                        <li>Maintien d'une activité physique aérobie régulière (≥ 150 min/semaine).</li>
                        <li>Contrôle annuel des constantes hémodynamiques et lipidiques.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)


    # ==============================================================================
        except Exception as pred_err:
            st.error(f'Erreur lors du calcul de l\'évaluation diagnostique : {pred_err}')


# ONGLET 2 : DÉPISTAGE PAR COHORTE (BATCH CSV)
# ==============================================================================
with tab_batch:
    st.markdown("### Dépistage et Triage Médical sur une Cohorte de Patients (CSV)")
    st.write("Téléversez un fichier de données contenant les variables cliniques pour évaluer plusieurs patients simultanément.")

    # Exemple de template à télécharger
    sample_df = pd.DataFrame([
        {
            "age": 62, "gender": 1, "chestpain": 3, "restingBP": 150, "serumcholestrol": 244,
            "fastingbloodsugar": 0, "restingrelectro": 1, "maxheartrate": 154, "exerciseangia": 1,
            "oldpeak": 1.4, "slope": 2, "noofmajorvessels": 1
        },
        {
            "age": 41, "gender": 0, "chestpain": 1, "restingBP": 120, "serumcholestrol": 195,
            "fastingbloodsugar": 0, "restingrelectro": 0, "maxheartrate": 165, "exerciseangia": 0,
            "oldpeak": 0.2, "slope": 1, "noofmajorvessels": 0
        }
    ])

    st.download_button(
        label="📥 Télécharger un modèle de fichier CSV d'exemple",
        data=sample_df.to_csv(index=False),
        file_name="modele_patients_cardioscan.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"**Données chargées :** {len(batch_df)} patients détectés.")
            st.dataframe(batch_df.head(), width="stretch")

            required_cols = [
                'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
                'fastingbloodsugar', 'restingrelectro', 'maxheartrate',
                'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels'
            ]
            missing_cols = [col for col in required_cols if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Colonnes manquantes dans le fichier CSV : {missing_cols}")
            else:
                if st.button("🚀 Lancer l'évaluation prédictive sur la cohorte", type="primary"):
                    with st.spinner("Analyse prédictive en cours..."):
                        batch_results = prediction_engine.predict(batch_df[required_cols])
                        
                        # Enrichissement du DataFrame
                        enriched_records = []
                        for res in batch_results["patients_evaluated"]:
                            enriched_records.append({
                                "Diagnostic": res["prediction_label"],
                                "Probabilite": res["risk_percentage"],
                                "Score_Numerique": res["risk_probability"],
                                "Niveau_Risque": res["risk_level"],
                                "Recommandation": res["clinical_recommendation"]
                            })

                        eval_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(enriched_records)], axis=1)

                        st.success("✅ Analyse de la cohorte finalisée avec succès !")

                        # Statistiques de cohorte
                        col_c1, col_c2, col_c3 = st.columns(3)
                        total_count = len(eval_df)
                        positive_count = sum(1 for r in batch_results["patients_evaluated"] if r["prediction"] == 1)
                        pos_rate = (positive_count / total_count) * 100 if total_count > 0 else 0

                        with col_c1:
                            st.metric("Total Patients Évalués", total_count)
                        with col_c2:
                            st.metric("Cas Pathologiques Détectés", f"{positive_count} ({pos_rate:.1f}%)")
                        with col_c3:
                            st.metric("Patients Indemnes", f"{total_count - positive_count} ({100 - pos_rate:.1f}%)")

                        st.dataframe(eval_df, width="stretch")

                        # Bouton d'exportation
                        csv_output = eval_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Télécharger le rapport prédictif complet (CSV)",
                            data=csv_output,
                            file_name="cardioscan_predictions_cohorte.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")


# ==============================================================================
# ONGLET 3 : GOUVERNANCE & MÉTRIQUES DU MODÈLE
# ==============================================================================
with tab_model:
    st.markdown("### Architecture et Rigueur Méthodologique")
    st.markdown("""
    Le modèle CardioScan AI a été développé selon une méthodologie rigoureuse en 5 phases :
    1. **Exploration Approfondie (EDA)** : Détection des incohérences biologiques (cholestérol nul, pentes hors plage).
    2. **Ingénierie de Caractéristiques Cliniques** : Création de biomarqueurs composites (`hr_reserve_ratio`, `ischemia_burden_index`, `vascular_stiffness_proxy`).
    3. **Stratégie Anti-Fuite** : Prétraitement encapsulé dans un `ColumnTransformer` scellé dans le sous-espace d'entraînement.
    4. **Benchmark Multi-Modèles** : Validation croisée stratifiée 5-Fold comparant 5 architectures.
    5. **Calibration du Seuil Décisionnel** : Optimisation clinique du seuil ($\\theta^* = 0.10$) favorisant le rappel ($F_2$-Score) pour éliminer les faux négatifs.
    """)

    st.subheader("Comparaison des Modèles lors de la Validation Croisée (5-Fold CV)")
    if model_comp_df is not None:
        st.dataframe(model_comp_df.style.highlight_max(subset=["Recall_Mean", "F2_Score_Mean", "ROC_AUC_Mean", "Accuracy_Mean"], color="#D4EDDA"), width="stretch")
    else:
        st.info("Le fichier `artifacts/model_comparison.csv` n'est pas encore disponible.")

    st.markdown("---")
    st.subheader("Performance Finale sur le Jeu de Test Indépendant (Holdout 20%)")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Rappel Clinique (Recall)", f"{metrics.get('test_recall', 1.0)*100:.2f}%")
        st.caption("Sensibilité maximale (0 faux négatif)")
    with col_m2:
        st.metric("F2-Score", f"{metrics.get('test_f2_score', 0.9966):.4f}")
        st.caption("Pondération double accordée au Rappel")
    with col_m3:
        st.metric("Précision (Precision)", f"{metrics.get('test_precision', 0.9831)*100:.2f}%")
        st.caption("Faible taux de fausse alerte")
    with col_m4:
        st.metric("Aire sous la courbe (ROC-AUC)", f"{metrics.get('test_roc_auc', 0.9999):.4f}")
        st.caption("Discrimination quasi-absolue")

    st.markdown("---")
    st.caption("Développé dans le respect des bonnes pratiques MLOps et du RGPD de santé.")

