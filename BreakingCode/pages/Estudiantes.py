import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# configuracion streamlit
st.set_page_config(page_title="Gestion de Estudiantes", page_icon="👥", layout="wide")

# carga y preproceso
@st.cache_data(show_spinner=False)
def load_datasets():
    folder = "datasets"
    est  = pd.read_csv(os.path.join(folder, "estudiantes.csv"))
    asig = pd.read_csv(os.path.join(folder, "asignaturas.csv"))
    rend = pd.read_csv(os.path.join(folder, "rendimiento.csv"))
    obs  = pd.read_csv(os.path.join(folder, "observaciones.csv"))

    # merge
    df = rend.merge(asig, left_on="asignatura", right_on="nombre_asignatura", how="left")
    df = df.merge(est, on="id_estudiante", how="left")

    # nota final
    periodos = ["P1", "P2", "P3", "P4"]
    df["nota"] = df["CF"].copy()
    if df["nota"].isna().any():
        df["nota"] = df[periodos].mean(axis=1)

    df["id_profesor"] = 0
    return df[["id_estudiante", "nombre_estudiante", "id_profesor", "aula", "asignatura", "nota"]], obs

# logica para obtener estudiantes
def get_student_by_id(df, sid):
    try:
        res = df[df["id_estudiante"] == int(sid)]
        return res if not res.empty else None
    except:
        return None

def list_all_students(df, obs):
    if df.empty:
        st.warning("No hay estudiantes registrados")
        return
    base = df.drop_duplicates("id_estudiante")[["id_estudiante", "nombre_estudiante", "aula"]]
    st.dataframe(base, use_container_width=True)

    # expandable con observaciones
    st.subheader("📝 Observaciones por estudiante")
    estudiante = st.selectbox("Seleccione estudiante (lista)", sorted(df["nombre_estudiante"].dropna().astype(str).unique()))
    sid = df[df["nombre_estudiante"] == estudiante]["id_estudiante"].iloc[0]
    obs_est = obs[obs["id_estudiante"] == sid]
    if not obs_est.empty:
        st.dataframe(obs_est[["fecha", "autor", "observacion"]], use_container_width=True)
    else:
        st.info("Sin observaciones para este estudiante")

def get_students_by_subject(df, subject):
    return df[df["asignatura"].str.contains(subject, case=False, na=False)]

def students_at_risk(df, threshold=60):
    notas = df["nota"] * 10 if df["nota"].max() <= 10 else df["nota"]
    return df.loc[notas < threshold, "id_estudiante"].tolist()

def get_database_stats(df):
    if df.empty:
        return None
    return dict(total_students=df["id_estudiante"].nunique(),
                average_score=df["nota"].mean(),
                highest_score=df["nota"].max(),
                lowest_score=df["nota"].min(),
                subjects=df["asignatura"].nunique())

def average_by_subject(df, subject):
    return df.loc[df["asignatura"] == subject, "nota"].mean()

# graficos de estudiante
def plot_student_skills(df, student_id):
    student = get_student_by_id(df, student_id)
    if student is None:
        st.error("Estudiante no encontrado")
        return
    name = student["nombre_estudiante"].iloc[0]
    data = df[df["nombre_estudiante"] == name]
    if data.empty:
        st.warning("Sin datos")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data.groupby("asignatura")["nota"].mean().sort_values().plot(kind="barh", ax=ax, color="skyblue")
    ax.set_xlabel("Nota")
    ax.set_title(f"Rendimiento por asignatura – {name}")
    st.pyplot(fig)

def plot_student_dashboard(df, student_id, obs):
    student = get_student_by_id(df, student_id)
    if student is None:
        return
    name = student["nombre_estudiante"].iloc[0]
    data = df[df["nombre_estudiante"] == name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Promedio", f"{data['nota'].mean():.2f}")
    c2.metric("🏆 Mejor nota", f"{data['nota'].max():.2f}")
    c3.metric("📉 Peor nota", f"{data['nota'].min():.2f}")
    c4.metric("📚 Asignaturas", data["asignatura"].nunique())

    # observaciones
    obs_est = obs[obs["id_estudiante"] == student_id]
    if not obs_est.empty:
        with st.expander("📝 Ver observaciones"):
            st.dataframe(obs_est[["fecha", "autor", "observacion"]], use_container_width=True)
    else:
        st.info("Sin observaciones para este estudiante")

    st.subheader("📋 Detalle de calificaciones")
    st.dataframe(data[["asignatura", "nota"]].sort_values("nota", ascending=False))

# sidebar para submodulos de estudiantes
df, obs = load_datasets()

option = st.sidebar.radio("📚 Listar:", ["📋 Ver todos", "📖 Por asignatura", "📊 Estadisticas", "📈 Graficos"])

if option == "📋 Ver todos":
    st.header("📋 Lista de estudiantes")
    list_all_students(df, obs)

elif option == "📖 Por asignatura":
    st.header("📖 Estudiantes por asignatura")
    subjects = sorted(df["asignatura"].dropna().astype(str).unique())
    subject = st.selectbox("Asignatura", subjects)
    tmp = get_students_by_subject(df, subject)
    st.dataframe(tmp.drop_duplicates("id_estudiante")[["id_estudiante", "nombre_estudiante", "aula"]])
    st.metric(f"📊 Promedio en {subject}", f"{average_by_subject(df, subject):.2f}")

elif option == "📊 Estadisticas":
    st.header("📊 Estadisticas generales")
    stats = get_database_stats(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Total estudiantes", stats["total_students"])
    c2.metric("📈 Promedio general", f"{stats['average_score']:.2f}")
    c3.metric("🏆 Nota max", f"{stats['highest_score']:.2f}")
    c4.metric("📉 Nota min", f"{stats['lowest_score']:.2f}")
    st.subheader("📊 Distribucion de notas")
    fig, ax = plt.subplots(figsize=(6, 3))
    df["nota"].plot(kind="hist", bins=15, ax=ax, color="steelblue", alpha=.7)
    st.pyplot(fig)

elif option == "📈 Graficos":
    st.header("📈 Graficos individuales")
    names = sorted(df["nombre_estudiante"].dropna().astype(str).unique())
    name = st.selectbox("Estudiante", names)
    sid = df[df["nombre_estudiante"] == name]["id_estudiante"].iloc[0]
    tipo = st.radio("Tipo", ["📊 Barras por asignatura", "🎯 Dashboard completo"])
    if tipo == "📊 Barras por asignatura":
        plot_student_skills(df, sid)
    else:
        plot_student_dashboard(df, sid, obs)

st.markdown("---")
st.caption("📊 Sistema de Gestion Estudiantil | Dataset 2025")