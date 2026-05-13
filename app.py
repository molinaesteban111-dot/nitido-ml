"""
NÍTIDO — App de Predicción de Candidatos
Acción 19 del Reto de Machine Learning II
Universidad Externado de Colombia
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NÍTIDO — Pre-filtrado IA",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 NÍTIDO · Sistema de Pre-filtrado de Candidatos")
st.markdown(
    "Ingresa las características del candidato y el modelo predecirá "
    "si avanza a entrevista, con una explicación SHAP local y, "
    "si es rechazado, un contrafactual accionable."
)

# ─────────────────────────────────────────
# Cargar modelo y objetos entrenados
# ─────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("model_nitido.pkl")
    scaler    = joblib.load("scaler_nitido.pkl")
    explainer = joblib.load("explainer_nitido.pkl")
    feat_cols = joblib.load("feature_cols.pkl")
    return model, scaler, explainer, feat_cols

try:
    model, scaler, explainer, FEAT_COLS = load_artifacts()
    artifacts_ok = True
except Exception as e:
    st.error(
        f"No se encontraron los archivos del modelo ({e}). "
        "Ejecuta primero `train_model.py` para generarlos."
    )
    artifacts_ok = False
    st.stop()

# ─────────────────────────────────────────
# Sidebar: Inputs del candidato
# ─────────────────────────────────────────
st.sidebar.header("📋 Datos del candidato")

def get_user_inputs():
    inputs = {}
    inputs["x1"]  = st.sidebar.slider("x1  (continua, 0–25)",     0.0, 25.0, 4.9,  0.1)
    inputs["x2"]  = st.sidebar.slider("x2  (continua, 7–100)",    7.0, 100.0, 64.7, 1.0)
    inputs["x3"]  = st.sidebar.selectbox("x3  (ordinal 1–5)",     [1, 2, 3, 4, 5], index=2)
    inputs["x4"]  = st.sidebar.selectbox("x4  (ordinal 0–8)",     list(range(9)),   index=4)
    inputs["x5"]  = st.sidebar.selectbox("x5  (binaria)",         [0, 1],           index=0)
    inputs["x6"]  = st.sidebar.slider("x6  (entero, 22–65)",      22,  65,  40,  1)
    inputs["x7"]  = st.sidebar.selectbox("x7  (ordinal 1–6)",     [1, 2, 3, 4, 5, 6], index=2)
    inputs["x8"]  = st.sidebar.selectbox("x8  (ordinal 1–10)",    list(range(1, 11)), index=4)
    inputs["x9"]  = st.sidebar.selectbox("x9  (ordinal 0–3)",     [0, 1, 2, 3],     index=1)
    inputs["x10"] = st.sidebar.slider("x10 (continua, 3.0–5.0)",  3.0, 5.0, 3.5,  0.01)
    inputs["x11"] = st.sidebar.slider("x11 (continua, 0–100)",    0.0, 100.0, 44.8, 0.1)
    inputs["x12"] = st.sidebar.selectbox("x12 (ordinal 0–11)",    list(range(12)),  index=5)
    inputs["x13"] = st.sidebar.selectbox("x13 (ordinal 0–3)",     [0, 1, 2, 3],     index=2)
    inputs["x14"] = st.sidebar.slider("x14 (continua, 0–24)",     0.0, 24.0, 2.0,  0.1)
    inputs["x15"] = st.sidebar.slider("x15 (continua, 1–50)",     1.0, 50.0, 9.0,  0.1)
    inputs["x16"] = st.sidebar.slider("x16 (continua, 0–100)",    0.0, 100.0, 55.0, 1.0)
    inputs["x17"] = st.sidebar.selectbox("x17 (binaria)",         [0, 1],           index=1)
    inputs["x18"] = st.sidebar.selectbox("x18 (ordinal 0–10)",    list(range(11)),  index=4)
    return inputs

inputs = get_user_inputs()
input_df = pd.DataFrame([inputs])

# ─────────────────────────────────────────
# Predicción
# ─────────────────────────────────────────
if st.button("🚀 Predecir", use_container_width=True):

    X_raw   = input_df[FEAT_COLS]
    X_scaled = scaler.transform(X_raw)

    proba  = model.predict_proba(X_scaled)[0]
    pred   = int(proba[1] >= 0.5)
    conf   = proba[pred]

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.success(f"✅ **APROBADO** para entrevista")
        else:
            st.error(f"❌ **RECHAZADO**")
    with col2:
        st.metric("Probabilidad de avanzar", f"{proba[1]:.1%}")

    # ─────────────────────────────────────
    # SHAP local
    # ─────────────────────────────────────
    st.subheader("📊 Explicación SHAP local")
    st.caption(
        "Las barras rojas empujan hacia rechazo (↓ probabilidad), "
        "las azules hacia aprobación (↑ probabilidad)."
    )

    shap_values = explainer(pd.DataFrame(X_scaled, columns=FEAT_COLS))

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ─────────────────────────────────────
    # Contrafactual (solo si rechazado)
    # ─────────────────────────────────────
    if pred == 0:
        st.divider()
        st.subheader("💡 Contrafactual: ¿Qué tendría que cambiar?")
        st.caption(
            "Se muestran las variables donde pequeñas mejoras aumentarían "
            "más la probabilidad de avanzar."
        )

        # Calculamos SHAP sobre el set original (sin escalar) para legibilidad
        sv = explainer(pd.DataFrame(X_scaled, columns=FEAT_COLS))
        shap_vals = sv[0].values  # shape (n_features,)

        # Variables que más perjudican (SHAP más negativo)
        importances = dict(zip(FEAT_COLS, shap_vals))
        worst_vars  = sorted(importances, key=lambda k: importances[k])[:5]

        cf_rows = []
        for var in worst_vars:
            curr_val = inputs[var]
            impact   = importances[var]
            # Sugerir aumentar o disminuir según signo del SHAP
            if impact < 0:
                # Aumentar el valor perjudica → sugerir dirección opuesta
                # Usamos heurística: si feature mean > current → ir hacia media
                col_mean = float(pd.read_csv(
                    "candidatos_nitido.csv"
                )[var].mean()) if True else curr_val
                direction = "↑ Aumentar" if col_mean > curr_val else "↓ Disminuir"
            else:
                direction = "— (ya aporta positivamente)"
            cf_rows.append({
                "Variable": var,
                "Valor actual": round(curr_val, 2),
                "Impacto SHAP": round(impact, 4),
                "Sugerencia": direction,
            })

        st.dataframe(pd.DataFrame(cf_rows), use_container_width=True)
        st.info(
            "⚠️ Estas sugerencias son orientativas. El sistema no puede garantizar "
            "aprobación con cambios aislados; múltiples factores interactúan."
        )

st.divider()
st.caption("NÍTIDO · MVP v1.0 · Universidad Externado de Colombia · ML II 2025")
