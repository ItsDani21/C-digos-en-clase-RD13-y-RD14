import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import pandas as pd
import numpy as np

def create_user_analysis(df, parent):
    """Crea un panel completo de análisis de géneros con scroll"""
    main_frame, scrollable_frame, canvas = _crear_frame_scrolleable(parent)
    create_user_grid_visualizations(df, scrollable_frame)
    return main_frame

def _crear_frame_scrolleable(parent):
    """Crea un frame principal con capacidad de scroll"""
    main_frame = tk.Frame(parent, bg="#0f0f23")
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(main_frame, bg="#0f0f23", highlightthickness=0)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    scrollable_frame = tk.Frame(canvas, bg="#0f0f23")
    canvas_frame = canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

    def configure_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", configure_scrollregion)

    def on_canvas_configure(event):
        canvas.itemconfig(canvas_frame, width=event.width)
    canvas.bind("<Configure>", on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    return main_frame, scrollable_frame, canvas

def _crear_frame_grafica(parent, titulo="", color_titulo="#00e5ff"):
    """Crea un frame contenedor para una gráfica con título"""
    frame = tk.Frame(parent, bg="#1a1a2e", relief="raised", bd=1)
    frame.pack(fill="x", padx=10, pady=10)
    
    if titulo:
        tk.Label(frame, text=titulo, font=("Arial",14,"bold"), 
                fg=color_titulo, bg="#1a1a2e").pack(pady=10)
    
    return frame

def _crear_grafico_top_generos_ventas(ax, df, colors):
    """Crea gráfico de top géneros por ventas"""
    if "genre" in df.columns and "total_sales" in df.columns:
        generos_por_ventas = (
            df.groupby("genre")["total_sales"]
            .sum()
            .reset_index()
            .rename(columns={"total_sales": "ventas_totales"})
            .sort_values(by="ventas_totales", ascending=True) 
            .head(5)
        )
        
        bars = ax.barh(range(len(generos_por_ventas)), 
                    generos_por_ventas["ventas_totales"], 
                    color=colors[:len(generos_por_ventas)])
        
        ax.set_yticks(range(len(generos_por_ventas)))
        ax.set_yticklabels(generos_por_ventas["genre"], color='white', fontsize=9)
        ax.set_title("Top 5 géneros con más ventas ", 
                    color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Ventas Totales (Millones)", color='white', fontsize=9)
        
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.2f}M", ha='left', va='center',
                    color='white', fontweight='bold', fontsize=8)

def _crear_grafico_preferencias_region(ax, df, colors):
    """Crea gráfico de preferencias de géneros por región"""
    if "genre" in df.columns and all(col in df.columns for col in ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']):
        ventas_por_genero = df.groupby("genre")[['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'total_sales']].sum()
        ventas_por_genero = ventas_por_genero.sort_values(by="total_sales", ascending=True)
        
        regions = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
        region_colors = ['#00e5ff', '#9c27b0', '#00ff88', '#ff6d00']
        region_labels = ['Norte América', 'Japón', 'PAL', 'Otras']
        
        left = np.zeros(len(ventas_por_genero))
        
        for i, region in enumerate(regions):
            bars = ax.barh(range(len(ventas_por_genero)), 
                        ventas_por_genero[region], 
                        left=left, 
                        color=region_colors[i],
                        label=region_labels[i],
                        alpha=0.8)
            left += ventas_por_genero[region].values
        
        ax.set_yticks(range(len(ventas_por_genero)))
        ax.set_yticklabels(ventas_por_genero.index, color='white', fontsize=9)
        ax.set_title("Preferencia de géneros por región", 
                    color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Ventas Totales (Millones)", color='white', fontsize=9)
        ax.legend(facecolor='#0a0a1a', edgecolor='#00e5ff', labelcolor='white', fontsize=8)
    
        for i, total in enumerate(ventas_por_genero['total_sales']):
            ax.text(total + 0.1, i, f"{total:.2f}M", ha='left', va='center',
                    color='white', fontweight='bold', fontsize=8)

def _crear_grafico_top_consolas_ventas(ax, df, colors):
    """Crea gráfico de top consolas por ventas por región"""
    if "console" in df.columns and all(col in df.columns for col in ['na_sales','jp_sales','pal_sales','other_sales','total_sales']):
        consolas_mas_ventas = (
            df.groupby("console")[['na_sales','jp_sales','pal_sales','other_sales','total_sales']]
            .sum()
            .reset_index()
            .sort_values(by="total_sales", ascending=False)
            .head(10)
        )
        
        bottom = [0] * len(consolas_mas_ventas)
        regiones = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
        region_labels = ['Norte América', 'Japón', 'PAL', 'Otras Regiones']
        region_colors = ['#00e5ff', '#9c27b0', '#00ff88', '#ff6d00']
        
        for i, region in enumerate(regiones):
            ax.bar(consolas_mas_ventas["console"], 
                    consolas_mas_ventas[region],
                    bottom=bottom, 
                    color=region_colors[i], 
                    label=region_labels[i],
                    alpha=0.9,
                    edgecolor='white',
                    linewidth=0.5)
            bottom = [sum(x) for x in zip(bottom, consolas_mas_ventas[region])]
        
        ax.set_title("🎮 Top 10 consolas con más Ventas por región", 
                    color='white', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel("Consola", color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel("Ventas Totales (Millones)", color='white', fontsize=11, fontweight='bold')
        
        ax.tick_params(axis='x', colors='white', labelrotation=45, labelsize=9)
        ax.tick_params(axis='y', colors='white', labelsize=9)
        
        for i, total in enumerate(consolas_mas_ventas["total_sales"]):
            ax.text(i, total + 0.5, f"{total:.2f}M", 
                    ha='center', va='bottom',
                    color='white', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8))
        
        ax.legend(facecolor='#1a1a2e', edgecolor='#00e5ff', 
                labelcolor='white', fontsize=10, loc='upper right')
        
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(2)
        
        ax.grid(True, alpha=0.2, color='#00e5ff')

def _crear_grafico_mejores_consolas(ax, df, colors):
    """Crea gráfico de consolas con mejores críticas"""
    if "console" in df.columns and "critic_score" in df.columns:
        consolas_scores = df.groupby('console')['critic_score'].mean().sort_values(ascending=False).head(8)
        
        bars = ax.bar(range(len(consolas_scores)), consolas_scores.values,
                    color=colors[:len(consolas_scores)], alpha=0.8)
        
        ax.set_title("Consolas con mejores Críticas promedio", 
                    color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel("Score Crítico Promedio", color='white')
        
        ax.set_xticks(range(len(consolas_scores)))
        ax.set_xticklabels(consolas_scores.index, rotation=45, ha='right', color='white')
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom',
                    color='white', fontweight='bold')
        
        ax.set_facecolor('#0a0a1a')

def _crear_grafico_mejores_publishers(ax, df, colors):
    """Crea gráfico de publishers con mejores calificaciones"""
    if "publisher" in df.columns and "critic_score" in df.columns:
        publishers_mejores_scores = (
            df.groupby("publisher")
            .agg({
                'critic_score': ['mean', 'count'],
                'total_sales': 'sum'
            })
            .round(2)
        )
        
        publishers_mejores_scores.columns = ['score_promedio', 'cantidad_juegos', 'ventas_totales']
        publishers_mejores_scores = publishers_mejores_scores[publishers_mejores_scores['cantidad_juegos'] >= 5]
        top_publishers = publishers_mejores_scores.nlargest(8, 'score_promedio')
        
        bars = ax.barh(range(len(top_publishers)), 
                    top_publishers['score_promedio'],
                    color=colors[:len(top_publishers)],
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=1)
        
        ax.set_title("Top publishers por calificación crítica", 
                    color='white', fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Score Crítico Promedio (0-10)", color='white', fontsize=10, fontweight='bold')
        ax.set_ylabel("Publisher", color='white', fontsize=10, fontweight='bold')
        
        ax.set_yticks(range(len(top_publishers)))
        ax.set_yticklabels(top_publishers.index, color='white', fontsize=9)
        
        for i, (idx, row) in enumerate(top_publishers.iterrows()):
            ax.text(row['score_promedio'] + 0.1, i, 
                    f"{row['score_promedio']}/10", 
                    ha='left', va='center',
                    color='white', fontweight='bold', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1a1a2e', alpha=0.8))
            
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(2)
        
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_xlim(0, 11)
        
        ax.axvline(x=7.5, color='#00ff88', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(7.5, len(top_publishers)-0.5, 'Alta Calidad', 
                color='#00ff88', fontsize=8, ha='center', va='top', fontweight='bold')

def _crear_grafico_top_juegos_genero(ax, df):
    """Crea gráfico de top juegos por género con promedios multiconsola"""
    if "title" in df.columns and "critic_score" in df.columns and "genre" in df.columns:
        df_unique = df.copy()

        if df_unique['critic_score'].max() > 10:
            df_unique['critic_score'] = df_unique['critic_score'] / 10

        df_unique = (
            df_unique.groupby(['title', 'genre'])
            .agg({
                'critic_score': 'mean',
                'console': lambda x: ', '.join(sorted(x.unique()))
            })
            .reset_index()
        )

        df_unique = df_unique.sort_values('critic_score', ascending=False)
        top_genres = df_unique['genre'].value_counts().head(5).index

        ax.axis('off')
        cell_height = 0.08
        start_y = 0.92

        for genre_idx, genre in enumerate(top_genres):
            ax.text(0.02, start_y, f"♦ {genre}",
                    transform=ax.transAxes, color='#00e5ff',
                    fontsize=11, fontweight='bold')

            start_y -= cell_height

            top_games = (
                df_unique[df_unique['genre'] == genre]
                .nlargest(3, 'critic_score')[['title', 'critic_score', 'console']]
            )

            medals = ['①', '②', '③']

            for game_idx, (idx, game) in enumerate(top_games.iterrows()):
                bg_color = '#1a1a2e' if game_idx % 2 == 0 else '#0f0f23'
                rect = plt.Rectangle((0.02, start_y - 0.02), 0.96, cell_height,
                                    transform=ax.transAxes, facecolor=bg_color, alpha=0.7)
                ax.add_patch(rect)

                ax.text(0.05, start_y, medals[game_idx],
                        transform=ax.transAxes, color='white', fontsize=10)

                ax.text(0.10, start_y, game['title'][:28],
                        transform=ax.transAxes, color='white', fontsize=9)

                ax.text(0.60, start_y, f"({game['console'][:20]}...)" if len(game['console']) > 23 else f"({game['console']})",
                        transform=ax.transAxes, color="#f3d739", fontsize=8)

                ax.text(0.85, start_y, f"{game['critic_score']:.1f}/10",
                        transform=ax.transAxes, color='#00ff88', fontsize=10, fontweight='bold')

                start_y -= cell_height

            start_y -= 0.02

        ax.text(0.5, 0.97, "✪ TOP 3 JUEGOS POR GÉNERO - PROMEDIO MULTICONSOLA",
                transform=ax.transAxes, color='white',
                fontsize=13, fontweight='bold', ha='center')

def _aplicar_estilo_graficos(axes):
    """Aplica estilo consistente a todos los gráficos"""
    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(1)
        ax.tick_params(colors='white', labelsize=8)

def _crear_primera_grafica(parent, df, available_width, colors):
    """Crea la primera figura con análisis de usuarios"""
    frame1 = _crear_frame_grafica(parent)

    fig1_width, fig1_height = min(available_width, 1400), min(available_width, 1400)*0.7
    
    fig1 = plt.figure(figsize=(fig1_width/100, fig1_height/100), facecolor='#0f0f23', dpi=100)
    gs1 = fig1.add_gridspec(2,2)

    ax1 = fig1.add_subplot(gs1[0,0:])
    ax2 = fig1.add_subplot(gs1[1,0:])

    _crear_grafico_mejores_consolas(ax1, df, colors)
    _crear_grafico_mejores_publishers(ax2, df, colors)

    _aplicar_estilo_graficos([ax1, ax2])
    fig1.tight_layout(pad=3.0)
    
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame1

def _crear_segunda_grafica(parent, df, available_width, colors):
    """Crea la segunda figura con análisis de críticas"""
    frame2 = _crear_frame_grafica(parent, " ", "#00ff88")
    
    fig2_width, fig2_height = min(available_width, 1400), min(available_width, 1400)*0.4
    
    fig2 = plt.figure(figsize=(fig2_width/100, fig2_height/100*2), facecolor='#0f0f23', dpi=100)
    gs2 = fig2.add_gridspec(4,2)


    ax6 = fig2.add_subplot(gs2[0:4, 0:2])

    _crear_grafico_top_juegos_genero(ax6, df)

    _aplicar_estilo_graficos([ax6])
    fig2.tight_layout(pad=3.0)
    
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame2

def create_user_grid_visualizations(df, parent):
    """Crea un grid de gráficas para el análisis de géneros"""
    plt.style.use('dark_background')
    parent.update_idletasks()
    available_width = max(parent.winfo_width() - 40, 1200)
    colors = ["#21929e", "#6e1f7c", "#21a768", "#bd5f18", "#b8a81d"]

    _crear_primera_grafica(parent, df, available_width, colors)
    _crear_segunda_grafica(parent, df, available_width, colors)