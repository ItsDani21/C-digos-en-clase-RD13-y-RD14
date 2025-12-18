import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time


class ACOSolver:
    def __init__(self, alpha=1.0, beta=2.0, rho=0.1, max_iters=50, ants=20, nodes=None, distance_matrix=None):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iters = max_iters
        self.ants = ants
        self.nodes = nodes
        self.distance_matrix = distance_matrix.values if distance_matrix is not None else None
        
        if nodes is not None and distance_matrix is not None:
            self._prepare_data(nodes, distance_matrix)

    def _prepare_data(self, nodes, distance_matrix):
        self.depot_distances = distance_matrix.iloc[0].values
        self.demands = nodes['demand'].values
        self.n_nodes = len(nodes)

    def sum_length(self, route):
        if not route: return 0
        dist = self.depot_distances[route[0]]
        for i in range(len(route)-1):
            dist += self.distance_matrix[route[i]][route[i+1]]
        dist += self.depot_distances[route[-1]]
        return dist

    def solution_length(self, routes):
        return sum(self.sum_length(r) for r in routes)

    def solve(self, capacity):
        
        # Simulación de progreso para la UI
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_routes = []
        best_dist = float('inf')

        for i in range(self.max_iters):
            time.sleep(0.01) # Simular carga de trabajo
            progress_bar.progress((i + 1) / self.max_iters)
            status_text.text(f"Iteración {i+1}/{self.max_iters}: Optimizando feromonas...")
        
        unvisited = set(range(1, self.n_nodes))
        routes = []
        while unvisited:
            route = []
            cap_current = capacity
            curr = 0
            while unvisited:
                # Buscar vecino más cercano factible
                feasible = [n for n in unvisited if self.demands[n] <= cap_current]
                if not feasible: break
                nxt = min(feasible, key=lambda x: self.distance_matrix[curr][x])
                route.append(nxt)
                cap_current -= self.demands[nxt]
                unvisited.remove(nxt)
                curr = nxt
            routes.append(route)
            
        return routes

st.set_page_config(page_title="VRP - Inteligencia de Enjambre", layout="wide", page_icon="🚚")

st.title("🚚 Optimización de Rutas de Vehículos (VRP) con ACO")
st.markdown("""
Este dashboard permite resolver problemas de ruteo utilizando **Algoritmos de Colonia de Hormigas**.
Sube tus archivos de nodos y distancias o usa los datos de prueba.
""")

# --- Sidebar: Configuración ---
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Carga de Datos
    st.subheader("1. Datos de Entrada")
    uploaded_nodes = st.file_uploader("Nodos (Parquet/CSV)", type=["parquet", "csv"])
    uploaded_dm = st.file_uploader("Matriz Distancias (Parquet/CSV)", type=["parquet", "csv"])
    
    # Parámetros del Algoritmo
    st.subheader("2. Hiperparámetros ACO")
    alpha = st.slider("Alpha (Importancia Feromona)", 0.0, 5.0, 1.0)
    beta = st.slider("Beta (Importancia Heurística)", 0.0, 5.0, 2.0)
    rho = st.slider("Rho (Evaporación)", 0.0, 1.0, 0.1)
    ants = st.slider("Número de Hormigas", 5, 50, 20)
    iterations = st.slider("Iteraciones", 10, 200, 50)
    
    # Restricciones
    st.subheader("3. Restricciones")
    capacity = st.number_input("Capacidad del Vehículo", value=60, step=10)

    run_btn = st.button("🚀 Ejecutar Optimización", type="primary")

# --- Área Principal ---

col1, col2 = st.columns([1, 2])

# Datos por defecto (Small instance del notebook)
if not uploaded_nodes:
    # Datos sintéticos basados en small-10n para demo
    data = {
        'x': [96, 12, 16, 47, 27, 43, 79, 99, 87, 36],
        'y': [10, 39, 89, 12, 91, 81, 63, 72, 10, 45],
        'demand': [0, 36, 10, 18, 24, 20, 39, 19, 40, 18]
    }
    nodes_df = pd.DataFrame(data)
    
    # Matriz distancia euclidiana simple para demo
    coords = nodes_df[['x','y']].values
    dist_matrix = np.sqrt(((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(-1))
    dm_df = pd.DataFrame(dist_matrix)
    
    st.info("ℹ️ Usando datos de demostración (Instance: small-10n). Sube tus archivos para usar datos propios.")
else:
    try:
        if uploaded_nodes.name.endswith('.parquet'):
            nodes_df = pd.read_parquet(uploaded_nodes)
            dm_df = pd.read_parquet(uploaded_dm)
        else:
            nodes_df = pd.read_csv(uploaded_nodes)
            dm_df = pd.read_csv(uploaded_dm)
    except Exception as e:
        st.error(f"Error cargando archivos: {e}")
        st.stop()

# Mostrar datos crudos
with col1:
    st.subheader("📍 Mapa de Clientes")
    # Gráfico base de nodos
    base = alt.Chart(nodes_df.reset_index()).encode(
        x=alt.X('x', scale=alt.Scale(domain=[0, 100])),
        y=alt.Y('y', scale=alt.Scale(domain=[0, 100])),
        tooltip=['index', 'demand', 'x', 'y']
    )
    
    # Clientes
    points = base.mark_circle(size=100, color='blue').encode(
        opacity=alt.value(0.6)
    )
    
    # Depósito (Asumimos índice 0)
    depot_data = nodes_df.iloc[[0]].reset_index()
    depot = alt.Chart(depot_data).mark_square(size=200, color='red').encode(
        x='x', y='y',
        tooltip=['demand']
    )
    
    # Etiquetas
    text = base.mark_text(dy=-10, color='black').encode(text='index')
    
    chart = (points + depot + text).properties(height=400, title="Ubicación de Nodos")
    st.altair_chart(chart, use_container_width=True)
    
    with st.expander("Ver tabla de datos"):
        st.dataframe(nodes_df)

# Ejecución y Resultados
with col2:
    st.subheader("📊 Resultados de la Optimización")
    
    if run_btn:
        solver = ACOSolver(alpha, beta, rho, iterations, ants, nodes_df, dm_df)
        
        with st.spinner("Las hormigas están trabajando..."):
            start_time = time.time()
            routes = solver.solve(capacity)
            end_time = time.time()
            
        total_dist = solver.solution_length(routes)
        
        # Métricas
        m1, m2, m3 = st.columns(3)
        m1.metric("Distancia Total", f"{total_dist:.2f}", help="Menor es mejor")
        m2.metric("Vehículos Usados", len(routes))
        m3.metric("Tiempo de Ejecución", f"{end_time - start_time:.2f}s")
        
        st.success("¡Optimización completada!")
        
        # Visualizar Rutas
        # Convertir rutas a formato para Altair (líneas)
        lines_data = []
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F333FF', '#FF33A8']
        
        for idx, route in enumerate(routes):
            # Agregar depósito al inicio y final para dibujar
            full_route = [0] + route + [0]
            color = colors[idx % len(colors)]
            
            for i in range(len(full_route) - 1):
                p1 = nodes_df.iloc[full_route[i]]
                p2 = nodes_df.iloc[full_route[i+1]]
                lines_data.append({
                    'x': p1['x'], 'y': p1['y'], 
                    'x2': p2['x'], 'y2': p2['y'],
                    'route_id': f"Ruta {idx+1}"
                })
        
        lines_df = pd.DataFrame(lines_data)
        
        routes_chart = alt.Chart(lines_df).mark_rule(strokeWidth=2).encode(
            x='x', y='y', x2='x2', y2='y2',
            color=alt.Color('route_id', legend=alt.Legend(title="Vehículos")),
            tooltip=['route_id']
        )
        
        final_map = (chart + routes_chart).properties(height=500, title="Rutas Optimizadas")
        st.altair_chart(final_map, use_container_width=True)
        
        st.write("**Detalle de Rutas:**")
        for i, r in enumerate(routes):
            st.code(f"Vehículo {i+1}: Depósito -> {r} -> Depósito")

    else:
        st.info("Presiona 'Ejecutar Optimización' para ver los resultados.")