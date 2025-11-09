import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_critic_sales_correlation(df):
    """
    Analiza la correlación entre las puntuaciones de la crítica (critic_score) y las ventas totales (total_sales).
    
    Parámetros:
    - df: DataFrame con los datos de juegos
    
    Columnas necesarias:
    - critic_score: calificación de la crítica
    - total_sales: ventas totales globales
    """
    # Procesar datos
    df_processed = df.copy()
    
    # Asegurarse de que las columnas necesarias existen
    required_columns = ['critic_score', 'total_sales']
    for col in required_columns:
        if col not in df_processed.columns:
            print(f"Error: Columna '{col}' no encontrada en el DataFrame")
            return None
    
    # Filtrar datos válidos (sin valores nulos en critic_score y total_sales)
    df_valid = df_processed.dropna(subset=['critic_score', 'total_sales'])
    
    # Crear figura con múltiples subplots para diferentes análisis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Correlación entre Crítica y Éxito Comercial', fontsize=18, fontweight='bold')
    
    # 1. Diagrama de dispersión con línea de tendencia
    sns.regplot(x='critic_score', y='total_sales', data=df_valid, scatter_kws={'alpha':0.4}, line_kws={'color':'red'}, ax=ax1)
    ax1.set_title('Relación entre Puntuación Crítica y Ventas Totales')
    ax1.set_xlabel('Puntuación Crítica')
    ax1.set_ylabel('Ventas Totales (millones)')
    
    # Calcular y mostrar el coeficiente de correlación y p-valor
    corr, p_value = stats.pearsonr(df_valid['critic_score'], df_valid['total_sales'])
    ax1.annotate(f'Correlación de Pearson: {corr:.2f}\nP-valor: {p_value:.4f}', 
                xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                fontsize=10, ha='left', va='top')
    
    # 2. Histograma de puntuaciones críticas con diferenciación de ventas
    # Categorizar ventas en alta/media/baja
    df_valid['sales_category'] = pd.qcut(df_valid['total_sales'], 3, labels=['Bajas', 'Medias', 'Altas'])
    
    for category, color in zip(['Bajas', 'Medias', 'Altas'], ['#FF9999', '#66B2FF', '#99FF99']):
        subset = df_valid[df_valid['sales_category'] == category]
        sns.histplot(subset['critic_score'], kde=True, ax=ax2, color=color, alpha=0.6, label=category)
    
    ax2.set_title('Distribución de Puntuaciones Críticas por Categoría de Ventas')
    ax2.set_xlabel('Puntuación Crítica')
    ax2.set_ylabel('Número de Juegos')
    ax2.legend()
    
    # 3. Gráfico de caja y bigotes de ventas agrupadas por rango de puntuación crítica
    df_valid['score_range'] = pd.cut(df_valid['critic_score'], 
                                   bins=[0, 5, 6, 7, 8, 9, 10], 
                                   labels=['0-5', '5-6', '6-7', '7-8', '8-9', '9-10'])
    
    sns.boxplot(x='score_range', y='total_sales', data=df_valid, ax=ax3, palette='viridis')
    ax3.set_title('Ventas por Rango de Puntuación Crítica')
    ax3.set_xlabel('Rango de Puntuación Crítica')
    ax3.set_ylabel('Ventas Totales (millones)')
    
    # Calcular y mostrar la media de ventas para cada rango
    medias = df_valid.groupby('score_range')['total_sales'].mean()
    for i, media in enumerate(medias):
        if not np.isnan(media):  # Verificar que hay datos para ese rango
            ax3.text(i, media, f'{media:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Heatmap de correlación de ventas y crítica por género o plataforma
    if 'genre' in df_valid.columns:
        pivot_column = 'genre'
        title_suffix = 'Género'
    elif 'console' in df_valid.columns:
        pivot_column = 'console'
        title_suffix = 'Plataforma'
    else:
        pivot_column = None
    
    if pivot_column:
        # Top 8 géneros/plataformas por número de juegos
        top_categories = df_valid[pivot_column].value_counts().head(8).index.tolist()
        corr_data = []
        
        for category in top_categories:
            subset = df_valid[df_valid[pivot_column] == category]
            if len(subset) > 5:  # Solo calcular si hay suficientes datos
                corr_val = np.corrcoef(subset['critic_score'], subset['total_sales'])[0, 1]
                avg_score = subset['critic_score'].mean()
                avg_sales = subset['total_sales'].mean()
                count = len(subset)
                corr_data.append({
                    'Categoría': category,
                    'Correlación': corr_val,
                    'Puntuación Media': avg_score,
                    'Ventas Medias': avg_sales,
                    'Número de Juegos': count
                })
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            corr_df = corr_df.sort_values('Correlación', ascending=False)
            
            # Crear un gráfico de barras con la correlación
            bars = ax4.bar(corr_df['Categoría'], corr_df['Correlación'], color='skyblue')
            ax4.set_title(f'Correlación entre Puntuación y Ventas por {title_suffix}')
            ax4.set_xlabel(title_suffix)
            ax4.set_ylabel('Coeficiente de Correlación')
            ax4.set_ylim(-1, 1)
            ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            # Añadir etiquetas de valor
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.1),
                        f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top')
    else:
        ax4.text(0.5, 0.5, 'No hay datos de género o plataforma disponibles', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Mostrar el gráfico
    plt.show()

# Función para mostrar el gráfico en una ventana Tkinter (opcional)
def show_critic_sales_correlation_in_tk(df, master):
    fig = plt.figure(figsize=(15, 12))
    create_critic_sales_correlation(df)
    canvas = FigureCanvasTkAgg(fig, master=master)
    return canvas