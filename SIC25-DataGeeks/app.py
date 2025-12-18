import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
)

st.set_page_config(page_title="HydroSense Dashboard", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
RISK_LABELS = {0: "Bajo", 1: "Medio", 2: "Alto"}

def build_classes_by_quartiles(df: pd.DataFrame, target_col: str = "FloodProbability") -> pd.DataFrame:
    q25, q50, q75 = df[target_col].quantile([0.25, 0.50, 0.75]).to_numpy()

    def flood_risk(p):
        if p <= q25:
            return 0
        elif p <= q75:
            return 1
        else:
            return 2

    out = df.copy()
    out["FloodRisk"] = out[target_col].apply(flood_risk)
    return out

def row_normalized_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums

def pretty_report_df(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(rep).T
    # Ordenar filas de clases al principio
    order = ["0", "1", "2", "accuracy", "macro avg", "weighted avg"]
    rep_df = rep_df.reindex([x for x in order if x in rep_df.index])
    return rep_df

def danger_level_from_prob(p_high: float) -> str:
    # Semáforo simple para el "aviso"
    if p_high >= 0.70:
        return "ALTO"
    if p_high >= 0.45:
        return "MEDIO"
    return "BAJO"

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def train_model(
    X_train, y_train,
    n_estimators: int,
    max_depth,
    min_samples_leaf: int,
    max_features,
    class_weight,
    random_state: int = 42,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# -----------------------------
# UI - Header
# -----------------------------
st.title("HydroSense — Panel interactivo de evaluación y predicción")
st.caption(
    "Dashboard para entrenar y evaluar un modelo RandomForest de riesgo (Bajo/Medio/Alto) y probar predicciones con probabilidades."
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("1) Cargar datos")

uploaded = st.sidebar.file_uploader("Sube tu dataset (CSV)", type=["csv"])

st.sidebar.header("2) Configuración del modelo")
n_estimators = st.sidebar.slider("n_estimators", 100, 1500, 400, 50)

max_depth_opt = st.sidebar.selectbox("max_depth", ["None", "8", "10", "12", "14", "16", "20"], index=3)
max_depth = None if max_depth_opt == "None" else int(max_depth_opt)

min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 80, 15, 1)

max_features_opt = st.sidebar.selectbox("max_features", ["sqrt", "log2", "0.5", "0.7", "1.0"], index=0)
if max_features_opt in ("sqrt", "log2"):
    max_features = max_features_opt
else:
    max_features = float(max_features_opt)

class_weight_opt = st.sidebar.selectbox("class_weight", ["None", "balanced", "balanced_subsample"], index=1)
class_weight = None if class_weight_opt == "None" else class_weight_opt

test_size = st.sidebar.slider("test_size", 0.10, 0.50, 0.20, 0.05)

st.sidebar.header("3) Umbral de alerta (clase Alto)")
high_risk_threshold = st.sidebar.slider("Umbral P(Alto)", 0.05, 0.95, 0.45, 0.01)

run_train = st.sidebar.button("Entrenar / Re-entrenar", type="primary")

# -----------------------------
# Main: Load / Prepare
# -----------------------------
if not uploaded:
    st.info("Sube un CSV (por ejemplo `flood.csv`) para empezar.")
    st.stop()

df_raw = load_csv(uploaded)

if "FloodProbability" not in df_raw.columns:
    st.error("El CSV debe incluir la columna `FloodProbability`.")
    st.stop()

df = build_classes_by_quartiles(df_raw, target_col="FloodProbability")

feature_cols = [c for c in df.columns if c not in ("FloodProbability", "FloodRisk")]
X = df[feature_cols]
y = df["FloodRisk"]

st.subheader("Vista rápida del dataset")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas", f"{len(df):,}")
c2.metric("Features", f"{len(feature_cols)}")
c3.metric("Target", "FloodRisk (0/1/2)")
c4.metric("Sin nulos", "Sí" if df[feature_cols + ["FloodProbability"]].isna().sum().sum() == 0 else "No")

with st.expander("Mostrar columnas y primeros registros", expanded=False):
    st.write("Columnas:", feature_cols + ["FloodProbability", "FloodRisk"])
    st.dataframe(df.head(20), use_container_width=True)

# Distribución de clases
st.subheader("Distribución de clases (FloodRisk)")
dist = y.value_counts(normalize=True).sort_index().rename(index=RISK_LABELS)
st.bar_chart(dist)

# Split fijo (para estabilidad)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=float(test_size),
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model (only when pressed OR first time)
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
if "trained" not in st.session_state:
    st.session_state.trained = False

if run_train or (st.session_state.model is None):
    with st.spinner("Entrenando modelo..."):
        model = train_model(
            X_train, y_train,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight
        )
    st.session_state.model = model
    st.session_state.trained = True

model = st.session_state.model

# -----------------------------
# Evaluation
# -----------------------------
st.subheader("Evaluación del modelo")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("Accuracy", f"{acc:.3f}")
m2.metric("Balanced accuracy", f"{bacc:.3f}")
m3.metric("Test size", f"{len(X_test):,} muestras")

report_df = pretty_report_df(y_test, y_pred)

colA, colB = st.columns([1.2, 1])

with colA:
    st.markdown("### Classification report (tabla)")
    st.dataframe(report_df, use_container_width=True)

with colB:
    st.markdown("### Matriz de confusión")
    labels = [0, 1, 2]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{RISK_LABELS[i]}" for i in labels],
        columns=[f"pred_{RISK_LABELS[i]}" for i in labels],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Matriz normalizada (por fila)")
    cmn = row_normalized_cm(cm)
    cmn_df = pd.DataFrame(
        cmn,
        index=[f"true_{RISK_LABELS[i]}" for i in labels],
        columns=[f"pred_{RISK_LABELS[i]}" for i in labels],
    )
    st.dataframe(cmn_df.style.format("{:.2f}"), use_container_width=True)

# Feature importance
st.subheader("Importancia de variables (RandomForest)")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
top_n = st.slider("Top N features", 5, min(30, len(feature_cols)), 15, 1)
st.dataframe(importances.head(top_n).to_frame("importance"), use_container_width=True)
st.bar_chart(importances.head(top_n))

# -----------------------------
# Interactive prediction
# -----------------------------
st.subheader("Predicción interactiva (simulación de una zona)")

st.caption(
    "Ajusta los valores de entrada. Se calcula la probabilidad por clase y se emite un aviso basado en P(Alto). "
    "Esto no es una predicción física de inundación; es un scoring de riesgo según el modelo entrenado."
)

# Construimos controles por feature usando min/max del dataset
with st.expander("Controles de entrada (features)", expanded=True):
    input_values = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        col_min = float(df[col_name].min())
        col_max = float(df[col_name].max())
        col_med = float(df[col_name].median())

        # Para features enteras discretas: slider int
        is_int_like = pd.api.types.is_integer_dtype(df[col_name]) or (
            np.all(np.mod(df[col_name].dropna().to_numpy(), 1) == 0)
        )

        target_col = cols[i % 3]
        if is_int_like:
            input_values[col_name] = target_col.slider(
                col_name, int(col_min), int(col_max), int(col_med), 1
            )
        else:
            input_values[col_name] = target_col.slider(
                col_name, col_min, col_max, col_med
            )

input_df = pd.DataFrame([input_values], columns=feature_cols)

proba = model.predict_proba(input_df)[0]  # orden por clase [0,1,2]
p_low, p_med, p_high = float(proba[0]), float(proba[1]), float(proba[2])

pred_class = int(np.argmax(proba))
pred_label = RISK_LABELS[pred_class]

st.markdown("### Resultado")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Predicción", f"{pred_label} ({pred_class})")
r2.metric("P(Bajo)", f"{p_low*100:.1f}%")
r3.metric("P(Medio)", f"{p_med*100:.1f}%")
r4.metric("P(Alto)", f"{p_high*100:.1f}%")

# Aviso basado en umbral para clase Alto
alert = "✅ Sin alerta crítica"
if p_high >= high_risk_threshold:
    alert = "⚠️ ALERTA: Riesgo alto probable"

st.markdown("### Aviso operativo")
st.write(f"**{alert}** — Umbral configurado: **{high_risk_threshold*100:.1f}%**")

level = danger_level_from_prob(p_high)
if level == "ALTO":
    st.error(f"Aviso: tienes aproximadamente un **{p_high*100:.1f}%** de probabilidad de **Riesgo Alto** según el modelo.")
elif level == "MEDIO":
    st.warning(f"Aviso: tienes aproximadamente un **{p_high*100:.1f}%** de probabilidad de **Riesgo Alto** según el modelo.")
else:
    st.success(f"Aviso: tienes aproximadamente un **{p_high*100:.1f}%** de probabilidad de **Riesgo Alto** según el modelo.")

st.markdown("### Nota")
st.write(
    "Este porcentaje es la probabilidad estimada por el clasificador para la clase **Alto** (P(Alto)). "
    "Puede usarse como señal de alerta temprana ajustando el umbral según el coste de falsos positivos vs falsos negativos."
)
