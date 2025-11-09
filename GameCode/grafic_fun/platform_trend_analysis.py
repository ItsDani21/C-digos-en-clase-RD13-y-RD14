import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_platform_quality_sales_trend(df):
    """
    Analiza la relación entre calidad (critic_score) y ventas totales (total_sales) 
    a lo largo del tiempo, separando los resultados por plataforma.
    
    Parámetros:
    - df: DataFrame con los datos de juegos
    
    Columnas necesarias:
    - release_date: fecha de lanzamiento
    - console: plataforma del juego
    - critic_score: calificación de la crítica
    - total_sales: ventas totales globales
    """
    # Procesar datos
    df_processed = df.copy()
    
    # Convertir release_date a datetime y extraer el año
    if 'release_date' in df_processed.columns:
        df_processed['release_date'] = pd.to_datetime(df_processed['release_date'])
        df_processed['year'] = df_processed['release_date'].dt.year
    elif 'Year' in df_processed.columns:
        df_processed['year'] = df_processed['Year']
    
    # Asegurarse de que las columnas necesarias existen
    required_columns = ['year', 'console', 'critic_score', 'total_sales']
    for col in required_columns:
        if col not in df_processed.columns:
            print(f"Error: Columna '{col}' no encontrada en el DataFrame")
            return None
    
    # Filtrar datos válidos (sin valores nulos en critic_score y total_sales)
    df_valid = df_processed.dropna(subset=['year', 'console', 'critic_score', 'total_sales'])
    
    # Seleccionar las principales plataformas (top 8 por número de juegos)
    top_platforms = df_valid['console'].value_counts().head(8).index.tolist()
    df_top = df_valid[df_valid['console'].isin(top_platforms)]
    
    # Agrupar datos por año y plataforma
    critic_by_year_platform = df_top.groupby(['year', 'console'])['critic_score'].mean().reset_index()
    sales_by_year_platform = df_top.groupby(['year', 'console'])['total_sales'].sum().reset_index()
    
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Tendencia de Calidad y Éxito Comercial por Plataforma', fontsize=16, fontweight='bold')
    
    # Colores para cada plataforma
    colors = plt.cm.tab10.colors
    platform_colors = {platform: colors[i % len(colors)] for i, platform in enumerate(top_platforms)}
    
    # 1. Evolución del puntaje crítico por plataforma
    for platform in top_platforms:
        platform_data = critic_by_year_platform[critic_by_year_platform['console'] == platform]
        ax1.plot(platform_data['year'], platform_data['critic_score'], 
                marker='o', linewidth=2, markersize=5, 
                label=platform, color=platform_colors[platform])
    
    ax1.set_title('Evolución del Puntaje Crítico por Plataforma')
    ax1.set_ylabel('Puntaje Crítico Promedio')
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Plataforma')
    
    # 2. Evolución de las ventas por plataforma
    for platform in top_platforms:
        platform_data = sales_by_year_platform[sales_by_year_platform['console'] == platform]
        ax2.plot(platform_data['year'], platform_data['total_sales'], 
                marker='o', linewidth=2, markersize=5, 
                label=platform, color=platform_colors[platform])
    
    ax2.set_title('Evolución de las Ventas por Plataforma')
    ax2.set_xlabel('Año')
    ax2.set_ylabel('Ventas Totales (millones)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(title='Plataforma')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Muestra la grafica y la retorna 
    plt.show()
    return fig

# Función para mostrar el gráfico en una ventana Tkinter (opcional)
def show_platform_quality_sales_trend_in_tk(df, master):
    fig = create_platform_quality_sales_trend(df)
    if fig:
        canvas = FigureCanvasTkAgg(fig, master=master)
        return canvas
    return None