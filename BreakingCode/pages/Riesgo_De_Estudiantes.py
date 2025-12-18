import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.data_loader import load_base_data
from core.nlp import cargar_modelo_nlp
from core.models_riesgo import calcular_riesgo


# config streamlit
st.set_page_config(
    page_title="Riesgo de Desercion",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Riesgo de Desercion Escolar")
st.caption("Factores académicos + análisis de observaciones (NLP)")


# carga de datos
@st.cache_data
def load_riesgo_data():
    data = load_base_data()

    est = data["est"]
    rend = data["rend"]
    obs = data["obs"]

    # --- nota promedio ---
    periodos = ["P1", "P2", "P3", "P4"]
    rend["nota"] = rend["CF"].fillna(rend[periodos].mean(axis=1))
    notas = rend.groupby("id_estudiante")["nota"].mean().reset_index(name="nota_promedio")

    # --- asistencia ---
    asist = rend.groupby("id_estudiante")["asistencia"].mean().reset_index(name="asistencia")

    # --- observaciones ---
    obs_est = (
        obs.groupby("id_estudiante")["observacion"]
        .apply(lambda x: " | ".join(x))
        .reset_index(name="observaciones")
    )

    # --- merge ---
    df = (
        est.merge(notas, on="id_estudiante", how="left")
           .merge(asist, on="id_estudiante", how="left")
           .merge(obs_est, on="id_estudiante", how="left")
    )

    return df


df_riesgo = load_riesgo_data()
modelo_nlp = cargar_modelo_nlp()


# calculo riesgo
def calcular_riesgos(df, modelo_nlp):
    df = df.copy()

    df[["Rd", "F"]] = df.apply(
        lambda row: pd.Series(calcular_riesgo(row, modelo_nlp)),
        axis=1
    )

    return df.sort_values("Rd", ascending=False).reset_index(drop=True)


df_riesgo = calcular_riesgos(df_riesgo, modelo_nlp)


# KPIs
c1, c2, c3 = st.columns(3)

c1.metric("Total estudiantes", len(df_riesgo))
c2.metric("Riesgo promedio", f"{df_riesgo['Rd'].mean():.1%}")
c3.metric("Alto riesgo (Rd > 0.6)", (df_riesgo["Rd"] > 0.6).sum())


# grafico top riesgo

st.subheader("📊 Estudiantes con mayor riesgo")

top = df_riesgo.head(5)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top["nombre_estudiante"], top["Rd"])
ax.set_xlabel("Riesgo de deserción (Rd)")
ax.set_title("Top 5 estudiantes con mayor riesgo")
ax.invert_yaxis()

st.pyplot(fig)


# tabla con detalles

st.subheader("📋 Detalle por estudiante")

df_show = df_riesgo.copy()
df_show["nota_promedio"] = df_show["nota_promedio"].round(1)

st.dataframe(
    df_show[
        ["id_estudiante", "nombre_estudiante",
         "nota_promedio", "asistencia", "F", "Rd"]
    ].style.format({
        "asistencia": "{:.1%}",
        "F": "{:.2f}",
        "Rd": "{:.1%}"
    }),
    use_container_width=True
)


# consulta por estudiante

st.subheader("🔎 Consulta individual")

names = (
    df_riesgo["nombre_estudiante"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
)

sel = st.selectbox("Selecciona un estudiante", names)

row = df_riesgo[df_riesgo["nombre_estudiante"] == sel].iloc[0]

st.info(
    f"**Rd:** {row['Rd']:.1%} | "
    f"**Nota:** {row['nota_promedio']:.1f} | "
    f"**Asistencia:** {row['asistencia']:.1%} | "
    f"**Ambiente (F):** {row['F']:.2f}"
)


# observaciones

obs_text = row["observaciones"]

if isinstance(obs_text, str) and obs_text.strip():
    with st.expander("📝 Ver observaciones"):
        for o in obs_text.split(" | "):
            st.write("• " + o)
else:
    st.info("Este estudiante no tiene observaciones registradas.")


st.markdown("---")
st.caption(
    "Modelo hibrido NLP + formula "
    "Rd = 0.25·(1-A) + 0.25·(max(0,75-N)/100) + 0.5·(1-F)"
)
