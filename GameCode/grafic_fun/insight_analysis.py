import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import pandas as pd
import numpy as np

def create_insight_analysis(df, parent):
    """Crea un panel completo de análisis de géneros con scroll"""
    main_frame, scrollable_frame, canvas = _crear_frame_scrolleable(parent)
    create_insight_grid_visualizations(df, scrollable_frame)
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

def _preparar_datos_analisis(df):
    """Prepara los datos para el análisis de insights"""
    juegos_promedio = (
        df.groupby('title')
        .agg({
            'critic_score': 'mean',
            'total_sales': 'mean',
            'genre': 'first',
            'console': lambda x: list(x.unique()),
            'publisher': 'first'
        })
        .round(2)
        .reset_index()
    )

    if juegos_promedio['critic_score'].max() > 10:
        juegos_promedio['critic_score'] = juegos_promedio['critic_score'] / 10

    conteo_generos = juegos_promedio.groupby("genre").size().reset_index(name='count')
    generos_filtrados = conteo_generos[conteo_generos['count'] > 10]['genre']
    df_filtrado = juegos_promedio[juegos_promedio['genre'].isin(generos_filtrados)]

    return df_filtrado

def _crear_grafico_calidad_ventas(ax, df_filtrado, colors):
    """Crea gráfico de Calidad vs Ventas"""
    if "critic_score" in df_filtrado.columns and "total_sales" in df_filtrado.columns:
        ax.scatter(
            df_filtrado['total_sales'], 
            df_filtrado['critic_score'],
            c='#00e5ff',
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )

        joyas_ocultas = df_filtrado[
            (df_filtrado['critic_score'] >= 8.0) & 
            (df_filtrado['total_sales'] < 0.5)
        ]
        blockbusters_vacios = df_filtrado[
            (df_filtrado['total_sales'] >= 2.0) & 
            (df_filtrado['critic_score'] < 6.0)
        ]

        if len(joyas_ocultas) > 0:
            ax.scatter(
                joyas_ocultas['total_sales'], 
                joyas_ocultas['critic_score'],
                c='#00ff88',
                s=80,
                edgecolors='white',
                linewidth=1.5,
                label='Joyas Ocultas'
            )

        ax.axhline(y=8.0, color='#00ff88', linestyle='--', alpha=0.5)
        ax.axhline(y=6.0, color='#ff6d00', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='#00ff88', linestyle='--', alpha=0.5)
        ax.axvline(x=2.0, color='#ff6d00', linestyle='--', alpha=0.5)

        ax.set_title("Calidad vs Ventas", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Ventas Totales (Millones)", color='white', fontsize=10)
        ax.set_ylabel("Puntaje Critico Promedio", color='white', fontsize=10)
        ax.legend(facecolor='#1a1a2e', edgecolor='#00e5ff', labelcolor='white', fontsize=9, loc='upper right')
        ax.set_xlim(-0.1, max(df_filtrado['total_sales']) * 1.1)
        ax.set_ylim(0, 10.5)
        
        return joyas_ocultas
    return pd.DataFrame()

def _crear_grafico_top_joyas(ax, joyas_ocultas):
    """Crea gráfico de Top 5 Joyas Ocultas"""
    ax.axis('off')

    if not joyas_ocultas.empty:
        top5 = joyas_ocultas.nlargest(5, 'critic_score')
        cell_height = 0.12
        start_y = 0.9

        ax.text(0.5, 0.97, "Top 5 Joyas Ocultas", transform=ax.transAxes,
                ha='center', color='white', fontsize=12, fontweight='bold')

        for idx, (_, row) in enumerate(top5.iterrows()):
            bg_color = '#1a1a2e' if idx % 2 == 0 else '#0f0f23'
            rect = plt.Rectangle((0.02, start_y - 0.04), 0.96, cell_height,
                                transform=ax.transAxes, facecolor=bg_color, alpha=0.7)
            ax.add_patch(rect)

            ax.text(0.05, start_y, f"{idx+1}. {row['title'][:25]}", transform=ax.transAxes,
                    color='white', fontsize=9, fontweight='bold')
            ax.text(0.55, start_y, f"{row['critic_score']:.1f}/10", transform=ax.transAxes,
                    color='#00ff88', fontsize=9, fontweight='bold')
            ax.text(0.75, start_y, f"{row['total_sales']:.2f}M", transform=ax.transAxes,
                    color='#00e5ff', fontsize=8)
            
            consolas = ", ".join(row['console'][:2])
            if len(row['console']) > 2:
                consolas += f" (+{len(row['console'])-2})"
            ax.text(0.9, start_y, consolas, transform=ax.transAxes,
                    color='#9c27b0', fontsize=8, ha='right')

            start_y -= cell_height
    else:
        ax.text(0.5, 0.5, "No se encontraron joyas ocultas", 
                transform=ax.transAxes, color='gray', fontsize=10, ha='center', va='center')

def _crear_grafico_analisis_publishers(ax, df):
    """Crea gráfico de análisis de publishers"""
    if all(col in df.columns for col in ['publisher', 'total_sales', 'console']):
        resumen_publishers = (
            df.groupby('publisher')
            .agg({
                'total_sales': 'sum',
                'title': 'count',
                'console': lambda x: len(set(x))
            })
            .rename(columns={'title': 'num_juegos', 'console': 'num_plataformas'})
            .reset_index()
        )

        top_publishers = resumen_publishers.sort_values('total_sales', ascending=False).head(10)

        ax.bar(top_publishers['publisher'], top_publishers['total_sales'],
                color='#00e5ff', alpha=0.7, label='Ventas Totales')
        
        ax3b = ax.twinx()
        ax3b.plot(top_publishers['publisher'], top_publishers['num_plataformas'],
                color='#ff6d00', marker='o', linewidth=2, label='Plataformas')

        ax.set_title("Top 10 publishers", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Publisher", color='white', fontsize=9)
        ax.set_ylabel("Ventas Totales (millones)", color='#00e5ff', fontsize=9)
        ax3b.set_ylabel("Plataformas", color='#ff6d00', fontsize=9)

        ax.tick_params(colors='white', rotation=40)
        ax3b.tick_params(colors='white')

        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#00e5ff', fontsize=8)
        ax3b.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#ff6d00', fontsize=8)

        for i, val in enumerate(top_publishers['total_sales']):
            ax.text(i, val + 1, f"{val:.1f}M", color='white', fontsize=8, ha='center', fontweight='bold')
            ax.text(i, val / 2, f"{top_publishers['num_juegos'].iloc[i]} juegos", 
                    color='#9c27b0', fontsize=7, ha='center')

def _crear_grafico_resumen_publishers(ax, df):
    """Crea gráfico de resumen ejecutivo de publishers"""
    ax.axis('off')
    
    # Estadísticas principales
    total_publishers = df['publisher'].nunique()
    top_publisher_ventas = df.groupby('publisher')['total_sales'].sum().idxmax()
    top_publisher_ventas_valor = df.groupby('publisher')['total_sales'].sum().max()
    top_publisher_juegos = df.groupby('publisher')['title'].count().idxmax()
    top_publisher_juegos_valor = df.groupby('publisher')['title'].count().max()
    
    # Publisher con mejor score promedio (mínimo 5 juegos)
    publishers_score = df.groupby('publisher').agg({
        'critic_score': 'mean',
        'title': 'count'
    })
    publishers_score = publishers_score[publishers_score['title'] >= 5]
    if not publishers_score.empty:
        mejor_publisher_score = publishers_score['critic_score'].idxmax()
        mejor_score_valor = publishers_score['critic_score'].max()
    else:
        mejor_publisher_score = "N/A"
        mejor_score_valor = 0

    start_y = 0.95
    cell_height = 0.08
    
    # Título
    ax.text(0.5, start_y, "Resumen- Publishers", 
            transform=ax.transAxes, ha='center', color='white', 
            fontsize=14, fontweight='bold')
    
    start_y -= 0.1
    
    # Estadísticas
    stats = [
        (f"Total publishers: {total_publishers}", '#00e5ff'),
        (f"Top ventas: {top_publisher_ventas[:20]} - {top_publisher_ventas_valor:.1f}M", '#00ff88'),
        (f"Más juegos: {top_publisher_juegos[:20]} - {top_publisher_juegos_valor} juegos", '#9c27b0'),
        (f"Mejor calidad: {mejor_publisher_score[:20]} - {mejor_score_valor:.1f}/10", '#ff6d00')
    ]
    
    for stat, color in stats:
        ax.text(0.05, start_y, stat, transform=ax.transAxes,
                color=color, fontsize=11, fontweight='bold')
        start_y -= cell_height
    
    # Top 5 publishers por diversidad de plataformas
    start_y -= 0.05
    ax.text(0.05, start_y, "Top 5 Publishers - Diversidad plataformas:", 
            transform=ax.transAxes, color='white', fontsize=10, fontweight='bold')
    start_y -= cell_height
    
    platform_diversity = df.groupby('publisher')['console'].nunique().nlargest(5)
    for i, (publisher, diversity) in enumerate(platform_diversity.items()):
        ax.text(0.08, start_y, f"{i+1}. {publisher[:25]} - {diversity} plataformas", 
                transform=ax.transAxes, color='#f3d739', fontsize=9)
        start_y -= cell_height * 0.8

def _crear_grafico_top_consolas_global(ax, df, colors):
    """Crea gráfico de top consolas global"""
    if "console" in df.columns and "total_sales" in df.columns:
        consolas_stats = df.groupby('console').agg({
            'total_sales': 'sum',
            'title': 'count',
            'critic_score': 'mean'
        }).round(2)
        
        top_consolas = consolas_stats.nlargest(8, 'total_sales')
        
        x = range(len(top_consolas))
        width = 0.35
        
        # Barras de ventas
        bars1 = ax.bar(x, top_consolas['total_sales'], width, 
                      color=colors[0], alpha=0.7, label='Ventas (M)')
        
        # Barras de cantidad de juegos (escala secundaria)
        ax2 = ax.twinx()
        bars2 = ax2.bar([i + width for i in x], top_consolas['title'], width,
                       color=colors[1], alpha=0.7, label='Juegos')
        
        ax.set_title("Top Consolas - Ventas y Catálogo", color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel("Consola", color='white', fontsize=9)
        ax.set_ylabel("Ventas Totales (Millones)", color=colors[0], fontsize=9)
        ax2.set_ylabel("Cantidad de Juegos", color=colors[1], fontsize=9)
        
        ax.set_xticks([i + width/2 for i in x])
        ax.set_xticklabels(top_consolas.index, rotation=45, ha='right', color='white')
        
        # Añadir valores
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.1,
                   f"{bar1.get_height():.1f}M", ha='center', va='bottom', 
                   color=colors[0], fontsize=8, fontweight='bold')
            ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                    f"{int(bar2.get_height())}", ha='center', va='bottom',
                    color=colors[1], fontsize=8, fontweight='bold')
        
        ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor=colors[0], fontsize=8)
        ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor=colors[1], fontsize=8)

def _crear_grafico_mejores_juegos_global(ax, df):
    """Crea gráfico de mejores juegos globales"""
    ax.axis('off')
    
    # Preparar datos
    df_prep = df.copy()
    if df_prep['critic_score'].max() > 10:
        df_prep['critic_score'] = df_prep['critic_score'] / 10
    
    mejores_juegos = df_prep.nlargest(8, 'critic_score')[['title', 'critic_score', 'genre', 'publisher']]
    
    start_y = 0.95
    cell_height = 0.1
    
    ax.text(0.5, start_y, "Top 8 Juegos - Mejor calificación", 
            transform=ax.transAxes, ha='center', color='white', 
            fontsize=12, fontweight='bold')
    
    start_y -= 0.08
    
    for idx, (_, row) in enumerate(mejores_juegos.iterrows()):
        bg_color = '#1a1a2e' if idx % 2 == 0 else '#0f0f23'
        rect = plt.Rectangle((0.02, start_y - 0.03), 0.96, cell_height,
                            transform=ax.transAxes, facecolor=bg_color, alpha=0.7)
        ax.add_patch(rect)
        
        ax.text(0.05, start_y, f"{idx+1}. {row['title'][:30]}", 
                transform=ax.transAxes, color='white', fontsize=9, fontweight='bold')
        ax.text(0.6, start_y, f"Score: {row['critic_score']:.1f}/10", 
                transform=ax.transAxes, color='#00ff88', fontsize=8)
        ax.text(0.8, start_y, f"Genero: {row['genre']}", 
                transform=ax.transAxes, color='#00e5ff', fontsize=7)
        ax.text(0.05, start_y - 0.04, f"Publisher: {row['publisher'][:25]}", 
                transform=ax.transAxes, color='#9c27b0', fontsize=7)
        
        start_y -= cell_height

def _aplicar_estilo_graficos(axes):
    """Aplica estilo consistente a todos los gráficos"""
    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(1)
        ax.tick_params(colors='white', labelsize=8)

def _crear_primera_grafica(parent, df, available_width, colors):
    """Crea la primera figura con análisis de insights"""
    frame1 = _crear_frame_grafica(parent, "")

    fig1_width, fig1_height = min(available_width, 1400), min(available_width, 1400)*0.7
    
    fig1 = plt.figure(figsize=(fig1_width/100, fig1_height/100), facecolor='#0f0f23', dpi=100)
    gs1 = fig1.add_gridspec(2, 2)

    df_filtrado = _preparar_datos_analisis(df)

    ax1 = fig1.add_subplot(gs1[0, 0])
    joyas_ocultas = _crear_grafico_calidad_ventas(ax1, df_filtrado, colors)

    ax2 = fig1.add_subplot(gs1[0, 1])
    _crear_grafico_top_joyas(ax2, joyas_ocultas)

    ax3 = fig1.add_subplot(gs1[1, :])
    _crear_grafico_analisis_publishers(ax3, df)

    _aplicar_estilo_graficos([ax1, ax3])
    fig1.tight_layout(pad=3.0)
    
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame1

def _crear_segunda_grafica(parent, df, available_width, colors):
    """Crea la segunda figura con resumen ejecutivo y tops globales"""
    frame2 = _crear_frame_grafica(parent, "Resumen  y Tops globales", "#00ff88")
    
    fig2_width, fig2_height = min(available_width, 1400), min(available_width, 1400)*0.8
    
    fig2 = plt.figure(figsize=(fig2_width/100, fig2_height/100), facecolor='#0f0f23', dpi=100)
    gs2 = fig2.add_gridspec(2, 2)

    ax4 = fig2.add_subplot(gs2[0, 0])
    _crear_grafico_resumen_publishers(ax4, df)

    ax5 = fig2.add_subplot(gs2[0, 1])
    _crear_grafico_top_consolas_global(ax5, df, colors)

    ax6 = fig2.add_subplot(gs2[1, 0])
    _crear_grafico_mejores_juegos_global(ax6, df)

    ax7 = fig2.add_subplot(gs2[1, 1])
    # Gráfico adicional: Distribución de géneros
    if "genre" in df.columns:
        top_genres = df['genre'].value_counts().head(6)
        ax7.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%',
               colors=colors, startangle=90, textprops={'color':'white'})
        ax7.set_title("Distribucion géneros", color='white', fontsize=11, fontweight='bold')

    _aplicar_estilo_graficos([ax5, ax7])
    fig2.tight_layout(pad=3.0)
    
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame2

def create_insight_grid_visualizations(df, parent):
    """Crea un grid de gráficas para el análisis de insights"""
    plt.style.use('dark_background')
    parent.update_idletasks()
    available_width = max(parent.winfo_width() - 40, 1200)
    colors = ['#00e5ff', '#9c27b0', '#00ff88', '#ff6d00', '#ffeb3b', '#2979ff']

    _crear_primera_grafica(parent, df, available_width, colors)
    _crear_segunda_grafica(parent, df, available_width, colors)