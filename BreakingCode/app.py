import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# configuracion streamlit
st.set_page_config(page_title="Dashboard Academico", layout="wide")

# carga de datos
@st.cache_data
def cargar_datos():
    folder = "datasets"
    est  = pd.read_csv(os.path.join(folder, "estudiantes.csv"))
    rend = pd.read_csv(os.path.join(folder, "rendimiento.csv"))
    obs  = pd.read_csv(os.path.join(folder, "observaciones.csv"))

    # nota y asistencia
    periodos = ["P1", "P2", "P3", "P4"]
    rend["nota"] = rend["CF"].fillna(rend[periodos].mean(axis=1))
    stats = rend.groupby("id_estudiante").agg(
        nota_promedio=("nota", "mean"),
        asistencia=("asistencia", "mean")
    ).reset_index()

    # cantidad de observaciones
    obs_count = obs["id_estudiante"].value_counts().reset_index()
    obs_count.columns = ["id_estudiante", "n_observaciones"]

    df = est.merge(stats, on="id_estudiante", how="left")\
            .merge(obs_count, on="id_estudiante", how="left")
    df["n_observaciones"] = df["n_observaciones"].fillna(0)

    return df

df = cargar_datos()

# KPIs
st.title("📊 Dashboard Academico")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📚 Total estudiantes", len(df))
col2.metric("📈 Promedio general", f"{df['nota_promedio'].mean():.1f}")
col3.metric("✅ Asistencia media", f"{df['asistencia'].mean():.1%}")
col4.metric("📝 Observaciones totales", df["n_observaciones"].sum())

# graficos generalizados de estudiantes
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribucion de notas")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(df["nota_promedio"], bins=15, color="skyblue", edgecolor="black")
    ax.set_xlabel("Nota promedio")
    ax.set_ylabel("Estudiantes")
    st.pyplot(fig)

with col2:
    st.subheader("📈 Asistencia vs Nota")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(df["asistencia"], df["nota_promedio"], alpha=0.6, color="teal")
    ax.set_xlabel("Asistencia (%)")
    ax.set_ylabel("Nota promedio")
    st.pyplot(fig)

# tabla resumen
st.subheader("📋 Resumen por estudiante")
st.dataframe(df[["id_estudiante", "nombre_estudiante", "nota_promedio", "asistencia", "n_observaciones"]]
             .style.format({"nota_promedio": "{:.1f}", "asistencia": "{:.1%}"}))

# selector individual
st.subheader("🔍 Detalle individual")
sel = st.selectbox("Seleccione estudiante", sorted(df["nombre_estudiante"].dropna().astype(str).unique()))
row = df[df["nombre_estudiante"] == sel].iloc[0]
st.write(f"**Nota:** {row['nota_promedio']:.1f} | **Asistencia:** {row['asistencia']:.1%} | **Observaciones:** {int(row['n_observaciones'])}")

st.markdown("---")
st.caption("📊 Dashboard Academico | Dataset 2025")
