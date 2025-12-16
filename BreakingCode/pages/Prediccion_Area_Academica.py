import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from core.data_loader import load_base_data
from core.nlp import cargar_modelo_nlp
from core.models_area import preparar_dataset, entrenar_modelo, calc_scores


# configuracion de streamlit

st.set_page_config(
    page_title="Recomendacion de Area Academica",
    page_icon="🔮",
    layout="wide"
)

st.title("🔮 Recomendacion de Areas Academicas")
st.caption("Basado en rendimiento academico, asistencia y ambiente emocional")


# carga de datos y modelos

@st.cache_resource
def load_all():
    data = load_base_data()
    modelo_nlp = cargar_modelo_nlp()

    df = preparar_dataset(
        est=data["est"],
        rend=data["rend"],
        obs=data["obs"],
        modelo_nlp=modelo_nlp
    )

    modelo_area = entrenar_modelo(df)

    return df, modelo_area, data["areas"]


df_full, modelo_area, areas_df = load_all()


# seleccion estudiante

st.subheader("🎓 Selecciona un estudiante")

names = (
    df_full["nombre_estudiante"]
    .dropna()
    .astype(str)
    .sort_values()
    .unique()
)

sel = st.selectbox("Estudiante", names)

row = df_full[df_full["nombre_estudiante"] == sel].iloc[0]


# metricas

c1, c2, c3 = st.columns(3)

c1.metric("Nota promedio", f"{row['nota_promedio']:.1f}")
c2.metric("Asistencia", f"{row['asistencia']:.1f}%")
c3.metric("Ambiente (F)", f"{row['F']:.2f}")


# prediccion de area

st.subheader("🎯 area recomendada")

sc, sn, ss = calc_scores(row["observacion"]) if isinstance(row["observacion"], list) else (0,0,0)
var_F = 0 

X_pred = pd.DataFrame([{
    "nota_promedio": row["nota_promedio"],
    "asistencia": row["asistencia"],
    "F": row["F"],
    "var_F": var_F,
    "edad": row.get("edad", 18),
    "num_obs": row["num_obs"],
    "score_ciencia": sc,
    "score_num": sn,
    "score_social": ss
}])


probs = modelo_area.predict_proba(X_pred)[0]

prob_df = (
    pd.DataFrame({
        "id_area": modelo_area.classes_,
        "probabilidad": probs
    })
    .merge(areas_df, on="id_area")
    .sort_values("probabilidad", ascending=False)
)

top = prob_df.head(3)

for i, r in top.iterrows():
    st.success(f"**{r['nombre_area']}** — {r['probabilidad']:.1%}")
    st.caption(r["descripcion"])

# probabilidades por areas

probs = modelo_area.predict_proba(X_pred)[0]

prob_df = (
    pd.DataFrame({
        "id_area": modelo_area.classes_,
        "probabilidad": probs
    })
    .merge(areas_df, on="id_area")
    [["nombre_area", "probabilidad"]]
    .sort_values("probabilidad", ascending=False)
)

st.write("### 📊 Probabilidad por area")
st.dataframe(
    prob_df.style.format({"probabilidad": "{:.2%}"}),
    use_container_width=True
)


# grafico radial de areas de estudio

st.write("### 🧭 Distribucion de probabilidades")

labels = prob_df["nombre_area"].tolist()
values = prob_df["probabilidad"].tolist()

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
values_c = values + values[:1]
angles_c = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(2.4, 2.4))
ax = fig.add_subplot(111, polar=True)

ax.plot(angles_c, values_c, linewidth=1)
ax.fill(angles_c, values_c, alpha=0.25)

ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=6)

ax.set_yticklabels([])
ax.grid(alpha=0.3)
ax.spines["polar"].set_visible(False)

ax.set_ylim(0, 1)

st.pyplot(fig, use_container_width=False)


# observaciones

st.subheader("📝 Observaciones del estudiante")

if isinstance(row["observacion"], list) and len(row["observacion"]) > 0:
    with st.expander("Ver observaciones"):
        for obs in row["observacion"]:
            st.write("• " + str(obs))
else:
    st.info("Este estudiante no tiene observaciones registradas.")


st.markdown("---")
st.caption("Modelo RandomForest + reglas academicas + analisis NLP")
