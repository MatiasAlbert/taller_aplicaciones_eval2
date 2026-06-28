# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD — Clasificador de Continuidad Empresarial (SII)
#  Solemne 2 · Taller de Aplicaciones · Magíster en Data Science (USS)
#
#  Ejecuta este dashboard con:
#       streamlit run app.py
#
#  Requiere el archivo 'modelo_clasificador.joblib', generado por el notebook
#  01_entrenamiento_clasificador.ipynb (deben estar en la misma carpeta).
# ─────────────────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

# ── Configuración general ────────────────────────────────────────────────────
st.set_page_config(page_title="Clasificador SII — Continuidad Empresarial",
                   page_icon="📊", layout="wide")

# Paleta corporativa (igual que en el notebook)
NAVY, BLUE, CYAN, GREEN, RED, GOLD = '#0A1E38', '#0047CC', '#00B4E6', '#276749', '#9B2335', '#B8962E'
# Ruta del modelo: SIEMPRE relativa a la carpeta donde está este app.py
# (así funciona aunque ejecutes `streamlit run` desde otra carpeta).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(BASE_DIR, "modelo_clasificador.joblib")


# ── Carga del modelo (en caché para no recargar en cada interacción) ─────────
@st.cache_resource
def cargar_bundle(ruta):
    return joblib.load(ruta)


if not os.path.exists(ART):
    st.error(
        f"No se encontró **modelo_clasificador.joblib**.\n\n"
        f"Se buscó en:\n\n`{ART}`\n\n"
        "Ejecuta el notebook `01_entrenamiento_clasificador.ipynb` y deja el `.joblib` "
        "en la **misma carpeta** que este `app.py` (`taller_aplicaciones`)."
    )
    st.stop()

B = cargar_bundle(ART)
FEAT = B['feat_clf']


# ── Encabezado ───────────────────────────────────────────────────────────────
st.title("📊 Clasificador de Continuidad Empresarial; Datos SII")
st.caption(
    "Random Forest balanceado (Solemne 1). Predice si un segmento de empresas "
    "**CONTINÚA** o **CESA** actividad al año siguiente. "
    "*Alcance: el resultado describe un segmento del SII, no una empresa individual.*"
)

tab_resultados, tab_probar = st.tabs(["📈 Resultados del modelo", "🧪 Probar el modelo"])


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — RESULTADOS DEL MODELO
# ═════════════════════════════════════════════════════════════════════════════
with tab_resultados:
    st.subheader("Desempeño en el conjunto de prueba (distribución real ≈81/19)")

    m = B['metrics']
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC-ROC",   f"{m['AUC-ROC']:.3f}")
    c2.metric("Accuracy",  f"{m['Accuracy']:.3f}")
    c3.metric("Precision", f"{m['Precision']:.3f}")
    c4.metric("Recall",    f"{m['Recall']:.3f}")
    c5.metric("F1-Score",  f"{m['F1-Score']:.3f}")

    st.caption(
        f"Entrenado con {B.get('n_train', '—')} casos balanceados (50/50) · "
        f"evaluado en {B.get('n_test', '—')} casos · años {B['year_min']}–{B['year_max']}."
    )
    st.divider()

    col_izq, col_der = st.columns(2)

    # Matriz de confusión
    with col_izq:
        st.markdown("**Matriz de confusión**")
        fig_cm, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(
            confusion_matrix=B['confusion_matrix'],
            display_labels=['Cesa', 'Continúa']
        ).plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title("Real vs. Predicho", color=NAVY, fontweight='bold')
        st.pyplot(fig_cm)

    # Curva ROC
    with col_der:
        st.markdown("**Curva ROC**")
        fig_roc, ax = plt.subplots(figsize=(4.5, 4))
        ax.plot(B['roc']['fpr'], B['roc']['tpr'], color=BLUE, lw=2,
                label=f"AUC = {B['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.6)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Capacidad discriminatoria", color=NAVY, fontweight='bold')
        ax.legend(loc="lower right")
        st.pyplot(fig_roc)

    st.divider()

    # Importancia de variables
    st.markdown("**Importancia de las variables (Random Forest)**")
    fi = pd.Series(B['feature_importances']).sort_values()
    fig_fi, ax = plt.subplots(figsize=(9, 4))
    ax.barh(fi.index, fi.values,
            color=[RED if v > 0.2 else BLUE for v in fi.values], edgecolor='white')
    for i, v in enumerate(fi.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va='center', fontsize=9)
    ax.set_xlabel("Importancia relativa")
    ax.set_title("¿Qué variables pesan más en la predicción?", color=NAVY, fontweight='bold')
    st.pyplot(fig_fi)
    st.caption(
        "Recordatorio: el modelo **no usa `año` directo** (obs. del profesor); "
        "lo reemplazan `post2015`, `post2020` y `decada`."
    )


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PROBAR EL MODELO (dato ingresado por pantalla)
# ═════════════════════════════════════════════════════════════════════════════
with tab_probar:
    st.subheader("Ingresa un segmento y clasifícalo")
    st.caption("Selecciona los atributos del segmento de empresas; el modelo estimará si continúa o cesa.")

    opt = B['category_options']  # nombres legibles de cada categoría

    col_a, col_b = st.columns(2)
    with col_a:
        rubro = st.selectbox("Rubro (CIIU)", opt['Rubro'])
        tramo_ventas = st.selectbox("Tramo según ventas", opt['Tramo según ventas'])
        genero = st.selectbox("Género asociado al RUT", opt['Género asociado al RUT'])
    with col_b:
        tramo_trab = st.selectbox("Tramo según trabajadores informados",
                                  opt['Tramo según trabajadores informados'])
        region = st.selectbox("Región", opt['Región'])
        año = st.slider("Año comercial", B['year_min'], B['year_max'], B['year_max'])

    n_emp = st.number_input("Número de empresas en el segmento", min_value=1, value=50, step=1)

    if st.button("🔮 Clasificar segmento", type="primary"):
        # 1) Codificar las categorías con los encoders guardados
        fila = {
            'sector_cod': B['encoders']['sector_cod'].transform([rubro])[0],
            'tamaño_cod': B['encoders']['tamaño_cod'].transform([tramo_ventas])[0],
            'genero_cod': B['encoders']['genero_cod'].transform([genero])[0],
            'trab_cod':   B['encoders']['trab_cod'].transform([tramo_trab])[0],
            'region_cod': B['encoders']['region_cod'].transform([region])[0],
            'n_emp':      n_emp,
            # 2) Features temporales derivadas del año (igual que en el entrenamiento)
            'post2015':   int(año >= 2015),
            'post2020':   int(año >= 2020),
            'decada':     (año - 2005) // 5,
        }
        # 3) DataFrame en el ORDEN exacto de las features + escalado guardado
        X_new = pd.DataFrame([[fila[c] for c in FEAT]], columns=FEAT)
        X_scaled = B['scaler'].transform(X_new)

        pred = int(B['model'].predict(X_scaled)[0])
        prob = float(B['model'].predict_proba(X_scaled)[0, 1])  # P(continúa)

        st.divider()
        if pred == 1:
            st.success(f"### ✅ CONTINÚA el año siguiente")
        else:
            st.error(f"### ⛔ CESA actividad el año siguiente")

        cprob1, cprob2 = st.columns(2)
        cprob1.metric("Probabilidad de continuar", f"{prob*100:.1f}%")
        cprob2.metric("Probabilidad de cesar", f"{(1-prob)*100:.1f}%")
        st.progress(prob)

        with st.expander("Ver el vector de entrada que recibió el modelo"):
            st.dataframe(X_new.T.rename(columns={0: "valor codificado"}))

        st.caption(
            "⚠️ Alcance: la predicción aplica a un **segmento agregado** del SII "
            "(combinación de atributos), no a una empresa concreta."
        )
