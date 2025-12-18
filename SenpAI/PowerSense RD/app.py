import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
import plotly.express as px
from datos_por_agno import provincias_por_region
from datos_por_agno import resumen_consumo_por_agno, resumen_consumo_por_region

# =========================
# Título
# =========================
st.markdown("""
<h1 style='font-size:32px; color:#1f77b4; text-align:center;'>
PowerSense RD: Tendencias de consumo eléctrico entre distribuidoras en República Dominicana
para optimizar operaciones empresariales
</h1>
""", unsafe_allow_html=True)

region = st.selectbox(
    "Selecciona una región:",
    ["Norte", "Sur", "Este"]
)
st.write(f"Has seleccionado la región: {region}")

medida = st.radio(
    "Selecciona una medida:",
    ["Energia", "Potencia"],
    horizontal=True
)
st.write(f"Medida seleccionada: {medida}")

# =========================
# Dataframe
# =========================
df_regiones_meses, df_regiones_clientes = resumen_consumo_por_region(region)
st.subheader(f"Consumo promedio por mes y tipo de cliente en la región {region}")


# st.subheader("Datos")
# st.dataframe(df, use_container_width=True)

# =========================
# Gráficos lado a lado
# =========================


fig_bar = px.bar(df_regiones_meses, x="Mes", y=medida, color="Mes",
                    text=medida, color_discrete_sequence=px.colors.qualitative.Vivid)
fig_bar.update_layout(
    xaxis_title="Mes",
    yaxis_title=medida,
    title="Gráfico de Barras",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_bar, use_container_width=True)

fig_pie = px.pie(df_regiones_clientes, names="Cliente", values=medida, color="Cliente",
                    color_discrete_sequence=px.colors.qualitative.Vivid, hole=0.3)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.update_layout(
    title="Gráfico de Pastel",
    margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig_pie, use_container_width=True)

# =========================
# Slider seleccion de agno
# =========================
agno = st.slider("Selecciona un año", 2012, 2024, 2024, 1)
st.write(f"Año seleccionado: {agno}")

# =========================
# Mapa de RD con datos por región
# =========================
st.subheader(f"Mapa de consumo de {medida} por región")

valores_por_region = resumen_consumo_por_agno(agno).set_index("region")[medida].to_dict()

# Cargar GeoJSON
with open("provinces_municipality_summary.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)


# Crear mapa con límites estrictos
m = folium.Map(
    location=[18.7357, -69.6],
    zoom_start=8,
    width=900,
    height=500,
    max_bounds=True,
    min_zoom=7 
)

# Función para obtener color según valor
def get_color(region):
    if region == "Norte":
        return "#FFA500"  # naranja
    elif region == "Este":
        return "#00C853"  # verde
    else:
        return "#D50000"  # rojo

for feature in geojson_data["features"]:
    provincia = feature["properties"]["province_name"]
    region = None
    for r, provincias in provincias_por_region.items():
        if provincia in provincias:
            region = r
            break
    
    color = get_color(region) if region else "#BDBDBD"  # gris si no tiene región
    
    folium.GeoJson(
        feature,
        style_function=lambda x, color=color: {
            "fillColor": color, "color": "black", "weight": 1, "fillOpacity": 0.6
        },
        tooltip=f"{region}: {valores_por_region[region]:.2f}" if region else provincia
    ).add_to(m)

st_folium(m, width=900, height=500)

# =========================
# Pie de página
# =========================
st.markdown("""
<hr style="margin-top:50px; border: none; height: 2px; background-color: #1f77b4;">

<div style='text-align:center; color:gray; font-size:14px;'>
Desarrollado con ⚡ por <span style='color:#1f77b4; font-weight:bold;'>SenpAI 先輩</span><br>
<em>Samsung Innovation Campus</em>
</div>
""", unsafe_allow_html=True)