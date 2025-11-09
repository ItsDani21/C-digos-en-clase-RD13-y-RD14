import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


def create_industry_evolution(df, parent):
    # Procesar datos - asumiendo que tienes columnas similares
    df_processed = df.copy()
    
    # Si no existe la columna year, crearla desde release_date o similar
    if 'year' not in df_processed.columns:
        if 'release_date' in df_processed.columns:
            df_processed['release_date'] = pd.to_datetime(df_processed['release_date'])
            df_processed['year'] = df_processed['release_date'].dt.year
        elif 'Year' in df_processed.columns:
            df_processed['year'] = df_processed['Year']
        else:
            # Si no hay año, usar índice como ejemplo
            df_processed['year'] = range(2000, 2000 + len(df))
    
    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evolución de la Industria de Videojuegos', fontsize=16, fontweight='bold')
    
    # 1. Ventas totales por año
    sales_by_year = df_processed.groupby('year')['total_sales'].sum().sort_index()
    ax1.plot(sales_by_year.index, sales_by_year.values, marker='o', linewidth=2, 
             color='#ff6b6b', markersize=6)
    ax1.set_title('Evolución de Ventas Totales')
    ax1.set_ylabel('Ventas Totales (millones)')
    ax1.set_xlabel('Año')
    ax1.grid(True, alpha=0.3)
    
    # 2. Puntuación crítica promedio por año
    if 'critic_score' in df_processed.columns:
        score_by_year = df_processed.groupby('year')['critic_score'].mean().sort_index()
        ax2.plot(score_by_year.index, score_by_year.values, marker='s', linewidth=2,
                 color='#4ecdc4', markersize=6)
        ax2.set_title('Evolución de la Calidad (Puntuación Crítica)')
        ax2.set_ylabel('Puntuación Promedio')
        ax2.set_xlabel('Año')
        ax2.grid(True, alpha=0.3)
    
    # 3. Número de juegos por año
    games_by_year = df_processed['year'].value_counts().sort_index()
    ax3.bar(games_by_year.index, games_by_year.values, color='#45b7d1', alpha=0.7)
    ax3.set_title('Número de Juegos por Año')
    ax3.set_ylabel('Cantidad de Juegos')
    ax3.set_xlabel('Año')
    
    # 4. Géneros más populares por año (heatmap)
    if 'genre' in df_processed.columns:
        # Top 5 géneros por año
        genre_year_pivot = pd.crosstab(df_processed['year'], df_processed['genre'])
        top_genres = genre_year_pivot.sum().nlargest(5).index
        genre_year_top = genre_year_pivot[top_genres]
        
        sns.heatmap(genre_year_top.T, ax=ax4, cmap='YlOrRd', annot=True, fmt='d')
        ax4.set_title('Géneros Más Populares por Año (Top 5)')
        ax4.set_xlabel('Año')
        ax4.set_ylabel('Género')
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)