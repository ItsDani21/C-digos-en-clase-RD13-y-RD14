import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Fraud Detector - Manual Input",
    page_icon="🚨",
    layout="wide"
)

# Estilos CSS
st.markdown("""
    <style>
    .big-font { font-size: 20px !important; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
    .stAlert { font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARGA Y ENTRENAMIENTO (CACHEADO) ---
# Esto se ejecuta solo una vez para crear el "cerebro" del modelo
@st.cache_resource
def train_model():
    try:
        # Buscamos el archivo localmente
        df = pd.read_parquet("0000.parquet")
    except FileNotFoundError:
        return None, None, None, None, None

    # Limpieza básica
    drop_cols = ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip',
                 'trans_num', 'unix_time', 'trans_date_trans_time', 'dob', 'Unnamed: 23', '6006']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.fillna(0)

    # Diccionarios para guardar los Encoders y las opciones únicas para el formulario
    encoders = {}
    unique_values = {}
    
    cat_cols = ['merchant', 'category', 'gender', 'job']
    for col in cat_cols:
        if col in df.columns:
            # Convertir a string para evitar errores
            df[col] = df[col].astype(str)
            
            # Guardamos valores únicos para los SelectBox
            unique_values[col] = sorted(df[col].unique().tolist())
            
            # Entrenamos el encoder
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Balanceo 1:1
    frauds = df[df['is_fraud'] == 1]
    non_frauds = df[df['is_fraud'] == 0].sample(n=len(frauds), random_state=42)
    df_balanced = pd.concat([frauds, non_frauds]).sample(frac=1, random_state=42)

    X = df_balanced.drop('is_fraud', axis=1)
    y = df_balanced['is_fraud']
    
    # Guardamos el orden de las columnas para replicarlo en la predicción
    feature_order = X.columns.tolist()

    # Escalador
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelo
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler, encoders, unique_values, feature_order

# --- CARGAR EL CEREBRO ---
with st.spinner("Cargando cerebro del modelo y entrenando con datos históricos..."):
    model, scaler, encoders, unique_values, feature_order = train_model()

# --- INTERFAZ DE USUARIO ---

st.title("🚨 Simulador de Fraude Bancario")
st.markdown("Introduce los datos de una transacción manual para analizarla.")

if model is None:
    st.error("❌ ERROR: No se encontró el archivo '0000.parquet' en la carpeta.")
    st.info("Por favor, coloca el archivo del dataset en el mismo directorio que este script y recarga la página.")
else:
    # Creamos dos columnas para el formulario
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Detalles de la Transacción")
        
        with st.form("prediction_form"):
            # Inputs Numéricos
            amt = st.number_input("Monto de la Transacción ($)", min_value=0.0, value=100.0, step=10.0)
            city_pop = st.number_input("Población de la Ciudad", min_value=0, value=50000, step=1000)
            
            # Inputs Categóricos (Usando los valores que aprendimos del dataset)
            category = st.selectbox("Categoría del Gasto", unique_values['category'])
            gender = st.selectbox("Género", unique_values['gender'])
            job = st.selectbox("Trabajo del Cliente", unique_values['job'])
            merchant = st.selectbox("Comerciante (Merchant)", unique_values['merchant'])
            
            st.markdown("---")
            st.markdown("**Datos Geográficos (Coordenadas)**")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitud Cliente", value=34.0)
            long = c2.number_input("Longitud Cliente", value=-80.0)
            
            c3, c4 = st.columns(2)
            merch_lat = c3.number_input("Latitud Comercio", value=34.5)
            merch_long = c4.number_input("Longitud Comercio", value=-80.5)

            # Botón de envío
            submitted = st.form_submit_button("🔍 ANALIZAR TRANSACCIÓN")

    # --- LÓGICA DE PREDICCIÓN ---
    with col_result:
        if submitted:
            st.subheader("📊 Resultado del Análisis")
            
            # 1. Construir el DataFrame con una sola fila
            input_data = {
                'merchant': [merchant],
                'category': [category],
                'amt': [amt],
                'gender': [gender],
                'city_pop': [city_pop],
                'job': [job],
                'lat': [lat],
                'long': [long],
                'merch_lat': [merch_lat],
                'merch_long': [merch_long]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # 2. Aplicar los mismos Encoders que usamos en el entrenamiento
            try:
                for col, le in encoders.items():
                    # Truco: Si el valor ingresado es nuevo (no visto en training), podría fallar.
                    # Aquí asumimos que los selectbox limitan a lo conocido.
                    input_df[col] = le.transform(input_df[col])
                
                # 3. Asegurar el orden de columnas
                input_df = input_df[feature_order]
                
                # 4. Escalar
                input_scaled = scaler.transform(input_df)
                
                # 5. Predecir
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1] # Probabilidad de ser clase 1 (Fraude)

                # 6. Mostrar Resultado
                if prediction == 1:
                    st.error("🚨 ¡ALERTA! POSIBLE FRAUDE DETECTADO")
                    st.metric("Probabilidad de Fraude", f"{probability:.2%}")
                    st.image("https://media.giphy.com/media/l2Je0oOcT4cioSIfu/giphy.gif", caption="Transacción sospechosa", width=300)
                else:
                    st.success("✅ TRANSACCIÓN SEGURA")
                    st.metric("Probabilidad de Fraude", f"{probability:.2%}")
                    st.balloons()
            
            except Exception as e:
                st.error(f"Ocurrió un error al procesar los datos: {e}")
        else:
            st.info("👈 Completa el formulario y presiona 'Analizar' para ver el resultado aquí.")
            st.image("https://cdn-icons-png.flaticon.com/512/2620/2620547.png", width=200, caption="Esperando datos...")