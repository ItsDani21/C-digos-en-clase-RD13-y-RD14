import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import pandas as pd

def crear_frame_scrolleable(parent):
    """Crea un frame principal con capacidad de scroll"""
    main_frame = tk.Frame(parent, bg="#0f0f23")
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    canvas = tk.Canvas(main_frame, bg="#0f0f23", highlightthickness=0)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    # Frame interno dentro del canvas
    scrollable_frame = tk.Frame(canvas, bg="#0f0f23")
    canvas_frame = canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

    # Configurar scroll automático
    def configure_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    scrollable_frame.bind("<Configure>", configure_scrollregion)

    # Ajustar ancho del frame interno al tamaño del canvas
    def on_canvas_configure(event):
        canvas.itemconfig(canvas_frame, width=event.width)
    canvas.bind("<Configure>", on_canvas_configure)

    # Scroll con la rueda del mouse
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    return main_frame, scrollable_frame, canvas

def crear_frame_grafica(parent, titulo, color_titulo="#00e5ff"):
    """Crea un frame contenedor para una gráfica con título"""
    frame = tk.Frame(parent, bg="#1a1a2e", relief="raised", bd=1)
    frame.pack(fill="x", padx=10, pady=10)
    
    if titulo:
        tk.Label(frame, text=titulo, font=("Arial",14,"bold"), 
                fg=color_titulo, bg="#1a1a2e").pack(pady=10)
    
    return frame

def crear_grafico_top_generos_barras(ax, df, colors):
    """Crea gráfico de barras horizontales de top géneros"""
    if "genre" in df.columns:
        top_genres = df["genre"].value_counts()
        bars = ax.barh(range(len(top_genres)), top_genres.values, color=colors)
        ax.set_yticks(range(len(top_genres)))
        ax.set_yticklabels(top_genres.index, color='white', fontsize=9)
        ax.set_title("∑ Cantidad de juegos por género", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Cantidad de Juegos", color='white', fontsize=9)
        
        for i, bar in enumerate(bars):
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                   f"{int(bar.get_width()):,}", ha='left', va='center',
                   color='white', fontweight='bold', fontsize=8)

def crear_grafico_distribucion_generos(ax, df, colors):
    """Crea gráfico de pie de distribución de géneros"""
    if "genre" in df.columns:
        top_10 = df["genre"].value_counts().head(6)
        wedges, texts, autotexts = ax.pie(top_10.values,
                                         labels=top_10.index,
                                         autopct='%1.1f%%',
                                         colors=colors,
                                         startangle=90,
                                         textprops={'color':'white','fontsize':8})
        for autotext in autotexts:
            autotext.set_fontweight('bold')
        ax.set_title("◕ Distribución de géneros en la muestra", color='white', fontsize=11, fontweight='bold')

def crear_grafico_ventas_region_genero(ax, df, colors):
    """Crea gráfico de ventas por región y género"""
    if "genre" in df.columns and all(col in df.columns for col in ['na_sales','jp_sales','pal_sales','other_sales']):
        # Agrupar ventas por género
        region_sales_by_genre = df.groupby('genre')[['na_sales','jp_sales','pal_sales','other_sales']].sum()
        
        bottom = [0]*len(region_sales_by_genre)
        regiones = ['na_sales','jp_sales','pal_sales','other_sales']
        
        for i, column in enumerate(regiones):
            ax.bar(region_sales_by_genre.index, region_sales_by_genre[column],
                  bottom=bottom, color=colors[i % len(colors)], label=column)
            bottom = [sum(x) for x in zip(bottom, region_sales_by_genre[column])]
        
        ax.set_title("Ventas por región y género", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Género", color='white', fontsize=9)
        ax.set_ylabel("Ventas (millones)", color='white', fontsize=9)
        ax.tick_params(colors='white', labelrotation=45)
        ax.legend(facecolor='#1a1a2e', edgecolor='#00e5ff', fontsize=8)
        ax.grid(True, alpha=0.3)

def crear_grafico_top_consolas_genero(ax, df, colors):
    """Crea gráfico de top consolas por género"""
    if "console" in df.columns and "genre" in df.columns:
        top_consoles = df['console'].value_counts().head(5).index
        top_genres = df['genre'].value_counts().head(5).index

        console_genre_data = []
        for console in top_consoles:
            console_df = df[df['console'] == console]
            genre_counts = console_df['genre'].value_counts()
            for genre in top_genres:
                console_genre_data.append({
                    'console': console,
                    'genre': genre,
                    'count': genre_counts.get(genre,0)
                })

        heatmap_data = pd.DataFrame(console_genre_data)
        pivot_data = heatmap_data.pivot(index='console', columns='genre', values='count').fillna(0)

        bottom = [0]*len(pivot_data)
        for i, genre in enumerate(pivot_data.columns):
            ax.bar(pivot_data.index, pivot_data[genre], bottom=bottom,
                  color=colors[i%len(colors)], label=genre)
            bottom = [sum(x) for x in zip(bottom, pivot_data[genre])]

        ax.set_title("✰ Top plataformas por género", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Consola", color='white', fontsize=9)
        ax.set_ylabel("Cantidad de Juegos", color='white', fontsize=9)
        ax.legend(facecolor='#1a1a2e', edgecolor='#00e5ff', fontsize=8)
        ax.grid(True, alpha=0.3)

def crear_grafico_publishers_genero(ax, df, colors):
    """Crea gráfico de análisis de publishers por género"""
    if "publisher" in df.columns and "genre" in df.columns:
        top_publishers = df['publisher'].value_counts().head(5).index
        pub_genre_data = df[df['publisher'].isin(top_publishers)].groupby(['publisher','genre']).size().unstack(fill_value=0)

        pub_genre_data.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(pub_genre_data.columns)])
        ax.set_title("Análisis de publishers por género", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Publisher", color='white', fontsize=9)
        ax.set_ylabel("Cantidad de Juegos", color='white', fontsize=9)
        ax.legend(facecolor='#1a1a2e', edgecolor='#00e5ff', fontsize=8)
        ax.grid(True, alpha=0.3)

def aplicar_estilo_graficos(axes):
    """Aplica estilo consistente a todos los gráficos"""
    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(1)
        ax.tick_params(colors='white', labelsize=8)

def crear_primera_grafica(parent, df, available_width):
    """Crea la primera figura con múltiples subgráficos"""
    frame1 = crear_frame_grafica(parent, "", "#00e5ff")
    
    fig1_width, fig1_height = min(available_width, 1400), min(available_width, 1400)*0.7
    colors = ["#10abbd", "#630c72", "#119256", "#a35418", "#bead0d"]
    
    fig1 = plt.figure(figsize=(fig1_width/100, fig1_height/100), facecolor='#0f0f23', dpi=100)
    gs1 = fig1.add_gridspec(2, 2)
    
    # Crear subgráficos
    ax1 = fig1.add_subplot(gs1[0, 0])  # Top géneros barras
    ax2 = fig1.add_subplot(gs1[0, 1])  # Distribución géneros pie
    ax3 = fig1.add_subplot(gs1[1, :])  # Ventas por región y género
    
    crear_grafico_top_generos_barras(ax1, df, colors)
    crear_grafico_distribucion_generos(ax2, df, colors)
    crear_grafico_ventas_region_genero(ax3, df, colors)
    
    aplicar_estilo_graficos([ax1, ax2, ax3])
    fig1.tight_layout(pad=3.0)
    
  
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame1

def crear_segunda_grafica(parent, df, available_width):
    """Crea la segunda figura con análisis específicos"""
    frame2 = crear_frame_grafica(parent, "", "#00ff88")
    
    fig2_width, fig2_height = min(available_width, 1400), min(available_width, 1400)*0.4
    colors = ['#00e5ff', '#9c27b0', '#00ff88', '#ff6d00', '#ffeb3b']
    
    fig2 = plt.figure(figsize=(fig2_width/100, fig2_height/100), facecolor='#0f0f23', dpi=100)
    gs2 = fig2.add_gridspec(1, 2)
    
    # Crear subgráficos
    ax4 = fig2.add_subplot(gs2[0, 0])  # Top consolas por género
    ax5 = fig2.add_subplot(gs2[0, 1])  # Publishers por género
    
    crear_grafico_top_consolas_genero(ax4, df, colors)
    crear_grafico_publishers_genero(ax5, df, colors)
    
    aplicar_estilo_graficos([ax4, ax5])
    fig2.tight_layout(pad=3.0)
    
    # Integrar en Tkinter
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame2

def crear_tercera_grafica(parent, df, available_width):
    """Crea la tercera figura con análisis adicionales de géneros"""
    frame3 = crear_frame_grafica(parent, "", "#ff6d00")
    
    fig3_width, fig3_height = min(available_width, 1400), min(available_width, 1400)*0.8
    colors = ['#ff6d00', '#2979ff', '#00c853', '#ff1744', '#9c27b0', '#ffeb3b', '#00e5ff', '#9c27b0']
    
    fig3 = plt.figure(figsize=(fig3_width/100, fig3_height/100), facecolor='#0f0f23', dpi=100)
    gs3 = fig3.add_gridspec(2, 2)
    
    # Crear subgráficos
    ax6 = fig3.add_subplot(gs3[0, 0])  # Top juegos por género
    ax7 = fig3.add_subplot(gs3[0, 1])  # Plataformas por género
    ax8 = fig3.add_subplot(gs3[1, 0])  # Score crítico por género
    ax9 = fig3.add_subplot(gs3[1, 1])  # Top publishers por género
    
    # Gráfico 6: Top juegos por género (mejores puntuados)
    crear_grafico_top_juegos_genero(ax6, df, colors)
    
    # Gráfico 7: Distribución de plataformas por género
    crear_grafico_plataformas_genero(ax7, df, colors)
    
    # Gráfico 8: Score crítico por género
    crear_grafico_score_genero(ax8, df, colors)
    
    # Gráfico 9: Top publishers por género
    crear_grafico_top_publishers_genero(ax9, df, colors)
    
    aplicar_estilo_graficos([ax6, ax7, ax8, ax9])
    fig3.tight_layout(pad=3.0)
    
    # Integrar en Tkinter
    canvas3 = FigureCanvasTkAgg(fig3, master=frame3)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame3

def crear_grafico_top_juegos_genero(ax, df, colors):
    """Crea gráfico de top juegos mejor puntuados por género"""
    if "title" in df.columns and "genre" in df.columns and "critic_score" in df.columns:
        # Encontrar el juego mejor puntuado de cada género
        top_games_by_genre = df.loc[df.groupby('genre')['critic_score'].idxmax()]
        top_games_by_genre = top_games_by_genre.nlargest(8, 'critic_score')
        
        bars = ax.barh(range(len(top_games_by_genre)), 
                      top_games_by_genre['critic_score'], 
                      color=colors[:len(top_games_by_genre)])
        
        ax.set_yticks(range(len(top_games_by_genre)))
        # Acortar títulos largos para mejor visualización
        labels = [f"{row['title'][:20]}...\n({row['genre']})" if len(row['title']) > 20 
                 else f"{row['title']}\n({row['genre']})" 
                 for _, row in top_games_by_genre.iterrows()]
        ax.set_yticklabels(labels, color='white', fontsize=8)
        ax.set_title("Top juegos por género (Score Crítico)", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Puntuación Crítica", color='white', fontsize=9)
        
        # Añadir valores en las barras
        for i, bar in enumerate(bars):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   f"{bar.get_width():.1f}", ha='left', va='center',
                   color='white', fontweight='bold', fontsize=8)

def crear_grafico_plataformas_genero(ax, df, colors):
    """Crea gráfico de distribución de plataformas por género"""
    if "console" in df.columns and "genre" in df.columns:
        # Top 5 géneros y top 5 plataformas
        top_genres = df['genre'].value_counts().head(5).index
        top_consoles = df['console'].value_counts().head(5).index
        
        # Filtrar datos
        filtered_df = df[df['genre'].isin(top_genres) & df['console'].isin(top_consoles)]
        
        if not filtered_df.empty:
            # Crear tabla de contingencia
            contingency = pd.crosstab(filtered_df['genre'], filtered_df['console'])
            
            # Gráfico de barras apiladas
            bottom = [0] * len(contingency)
            for i, console in enumerate(contingency.columns):
                ax.bar(contingency.index, contingency[console], 
                      bottom=bottom, color=colors[i % len(colors)], 
                      label=console, alpha=0.8)
                bottom = [sum(x) for x in zip(bottom, contingency[console])]
            
            ax.set_title("Plataformas por Género", color='white', fontsize=11, fontweight='bold')
            ax.set_xlabel("Género", color='white', fontsize=9)
            ax.set_ylabel("Cantidad de Juegos", color='white', fontsize=9)
            ax.tick_params(axis='x', rotation=45, colors='white')
            ax.legend(facecolor='#1a1a2e', edgecolor='#ff6d00', fontsize=7)
            ax.grid(True, alpha=0.3)

def crear_grafico_score_genero(ax, df, colors):
    """Crea gráfico de distribución de scores por género"""
    if "genre" in df.columns and "critic_score" in df.columns:
        # Top 6 géneros por cantidad
        top_genres = df['genre'].value_counts().head(6).index
        filtered_df = df[df['genre'].isin(top_genres)]
        
        # Crear boxplot
        genre_data = [filtered_df[filtered_df['genre'] == genre]['critic_score'] 
                     for genre in top_genres]
        
        box_plot = ax.boxplot(genre_data, labels=top_genres, patch_artist=True)
        
        # Colorear las cajas
        for i, box in enumerate(box_plot['boxes']):
            box.set(facecolor=colors[i % len(colors)], alpha=0.7)
        
        for median in box_plot['medians']:
            median.set(color='white', linewidth=2)
        
        ax.set_title("Distribución de críticas por género", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Género", color='white', fontsize=9)
        ax.set_ylabel("Score Crítico", color='white', fontsize=9)
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, alpha=0.3)

def crear_grafico_top_publishers_genero(ax, df, colors):
    """Crea gráfico de top publishers por género"""
    if "publisher" in df.columns and "genre" in df.columns and "total_sales" in df.columns:
        # Encontrar el publisher más exitoso por género (por ventas)
        top_publishers_by_genre = df.loc[df.groupby('genre')['total_sales'].idxmax()]
        top_publishers_by_genre = top_publishers_by_genre.nlargest(8, 'total_sales')
        
        # Gráfico de dispersión o barras
        genres = top_publishers_by_genre['genre']
        sales = top_publishers_by_genre['total_sales']
        publishers = top_publishers_by_genre['publisher']
        
        bars = ax.bar(range(len(genres)), sales, color=colors[:len(genres)], alpha=0.8)
        
        ax.set_xticks(range(len(genres)))
        # Crear labels combinando género y publisher (abreviado)
        labels = [f"{genre}\n({publisher[:12]}...)" if len(publisher) > 12 else f"{genre}\n({publisher})"
                 for genre, publisher in zip(genres, publishers)]
        ax.set_xticklabels(labels, color='white', fontsize=7, rotation=45)
        ax.set_title("Mejor publisher por género (Ventas)", color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel("Ventas Totales (millones)", color='white', fontsize=9)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}M', ha='center', va='bottom',
                   color='white', fontweight='bold', fontsize=7)
        
        ax.grid(True, alpha=0.3)
















def create_genre_analysis(df, parent):
    """Función principal que crea el panel completo de análisis de géneros"""
    # Configurar estilo de matplotlib
    plt.style.use('dark_background')
    
    # Crear frame scrolleable
    main_frame, scrollable_frame, canvas = crear_frame_scrolleable(parent)
    
    # Calcular ancho disponible
    scrollable_frame.update_idletasks()
    available_width = max(scrollable_frame.winfo_width() - 40, 1200)
    
    # Crear gráficas
    crear_primera_grafica(scrollable_frame, df, available_width)
    crear_segunda_grafica(scrollable_frame, df, available_width)
    crear_tercera_grafica(scrollable_frame,df,available_width)
    return main_frame