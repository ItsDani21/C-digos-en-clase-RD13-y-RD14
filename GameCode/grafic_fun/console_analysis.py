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

def crear_frame_grafica(parent, titulo="", color_titulo="#00e5ff"):
    """Crea un frame contenedor para una gráfica con título"""
    frame = tk.Frame(parent, bg="#1a1a2e", relief="raised", bd=1)
    frame.pack(fill="x", padx=10, pady=10)
    
    if titulo:
        tk.Label(frame, text=titulo, font=("Arial",14,"bold"), 
                fg=color_titulo, bg="#1a1a2e").pack(pady=10)
    
    return frame

def crear_grafico_top_consolas_juegos(ax, df, colors):
    """Crea gráfico de top consolas con más juegos"""
    if "console" in df.columns and "title" in df.columns:
        # Calcular consolas con más juegos
        consolas_mas_juegos = (
            df.groupby("console")["title"]
            .count()
            .reset_index()
            .rename(columns={"title": "num_juegos"})
            .sort_values(by="num_juegos", ascending=False)
        )
        
        top_consolas_juegos = consolas_mas_juegos.head(10)
        
        bars = ax.bar(range(len(top_consolas_juegos)), 
                     top_consolas_juegos["num_juegos"], 
                     color=colors[:len(top_consolas_juegos)],
                     alpha=0.8)
        
        ax.set_xticks(range(len(top_consolas_juegos)))
        ax.set_xticklabels(top_consolas_juegos["console"], 
                          color='white', fontsize=8, rotation=45, ha='right')
        ax.set_title("Top 10 consolas con más juegos", color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel("Número de Juegos", color='white', fontsize=10)
        ax.set_xlabel("Consola", color='white', fontsize=10)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f"{int(height):,}", ha='center', va='bottom',
                   color='white', fontweight='bold', fontsize=9)

def crear_grafico_ventas_totales(ax, df, colors):
    """Crea gráfico de ventas totales por consola"""
    if "console" in df.columns and "total_sales" in df.columns:
        total_sales_by_console = df.groupby('console')['total_sales'].sum().sort_values(ascending=False).head(10)
        total_sales_by_console.head(8).plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title(' Ventas Totales registradas por consola', color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel('Ventas (Millones)', color='white', fontsize=9)
        ax.tick_params(axis='x', rotation=45, colors='white', labelsize=8)
        ax.tick_params(axis='y', colors='white', labelsize=8)

def crear_grafico_puntuacion_promedio(ax, df, colors):
    """Crea gráfico de puntuación promedio por consola"""
    if "console" in df.columns and "critic_score" in df.columns:
        avg_score_by_console = df.groupby('console')['critic_score'].mean().sort_values(ascending=False).head(10)
        avg_score_by_console.head(8).plot(kind='barh', ax=ax, color=colors, alpha=0.8)
        ax.set_title('℗ Puntuación Promedio', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Puntuación', color='white', fontsize=9)  
        ax.set_ylabel('Consola', color='white', fontsize=9)     
        ax.tick_params(axis='x', colors='white', labelsize=8)  
        ax.tick_params(axis='y', colors='white', labelsize=8)

def crear_grafico_diversidad_generos(ax, df, colors):
    """Crea gráfico de donut de diversidad de géneros por plataforma"""
    if "console" in df.columns and "genre" in df.columns:
        genres_per_console = df.groupby('console')['genre'].nunique()
        top_consoles = genres_per_console.nlargest(8)

        wedges, texts, autotexts = ax.pie(
            top_consoles.values, 
            labels=top_consoles.index, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='white'),
            textprops={'color': 'white', 'fontsize': 8}
        )

        # Mejorar etiquetas
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        ax.set_title('◎ Diversidad de géneros por plataforma', color='white', fontsize=11, fontweight='bold', pad=20)

def aplicar_estilo_graficos(axes):
    """Aplica estilo consistente a todos los gráficos"""
    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(1)
        ax.tick_params(colors='white', labelsize=8)

def crear_primera_grafica_consolas(parent, df, available_width):
    """Crea la primera figura con análisis de consolas"""
    frame1 = crear_frame_grafica(parent, "", "#00e5ff")
    
    fig1_width, fig1_height = min(available_width, 1400), min(available_width, 1400)*0.7
    colors = ["#0d7a86", "#550663", "#0d9e5b", "#7c3f10", "#ad9e11"]
    
    fig1 = plt.figure(figsize=(fig1_width/100, fig1_height/100), facecolor='#0f0f23', dpi=100)
    gs1 = fig1.add_gridspec(2, 2)
    
    # Crear subgráficos
    ax1 = fig1.add_subplot(gs1[0, 0])  # Top consolas por juegos
    ax2 = fig1.add_subplot(gs1[0, 1])  # Ventas totales
    ax3 = fig1.add_subplot(gs1[1, 0])  # Puntuación promedio
    ax4 = fig1.add_subplot(gs1[1, 1])  # Diversidad de géneros
    
    crear_grafico_top_consolas_juegos(ax1, df, colors)
    crear_grafico_ventas_totales(ax2, df, colors)
    crear_grafico_puntuacion_promedio(ax3, df, colors)
    crear_grafico_diversidad_generos(ax4, df, colors)
    
    aplicar_estilo_graficos([ax1, ax2, ax3, ax4])
    fig1.tight_layout(pad=3.0)
    
    # Integrar en Tkinter
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1)
    canvas1.draw()
    canvas1.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame1






def crear_segunda_grafica_consolas(parent, df, available_width):
    """Crea la segunda figura con análisis avanzados de consolas"""
    frame2 = crear_frame_grafica(parent, "", "#00ff88")
    
    fig2_width, fig2_height = min(available_width, 1400), min(available_width, 1400)*0.8
    colors = ['#00ff88', '#ff6d00', '#2979ff', '#9c27b0', '#ffeb3b', '#00e5ff', '#ff1744', '#00c853']
    
    fig2 = plt.figure(figsize=(fig2_width/100, fig2_height/100), facecolor='#0f0f23', dpi=100)
    gs2 = fig2.add_gridspec(2, 2)
    
    # Crear subgráficos
    ax5 = fig2.add_subplot(gs2[0, 0])  # Ventas por región
    ax6 = fig2.add_subplot(gs2[0, 1])  # Top publishers por consola
    ax7 = fig2.add_subplot(gs2[1,0:])  # Evolución temporal
    
    crear_grafico_ventas_region_consola(ax5, df, colors)
    crear_grafico_top_publishers_consola(ax6, df, colors)
    crear_grafico_generos_exitosos_consola(ax7, df, colors)
    
    aplicar_estilo_graficos([ax5, ax6, ax7])
    fig2.tight_layout(pad=3.0)
    
    # Integrar en Tkinter
    canvas2 = FigureCanvasTkAgg(fig2, master=frame2)
    canvas2.draw()
    canvas2.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
    
    return frame2

def crear_grafico_ventas_region_consola(ax, df, colors):
    """Crea gráfico de ventas por región y consola"""
    if "console" in df.columns and all(col in df.columns for col in ['na_sales','jp_sales','pal_sales','other_sales']):
        # Top 6 consolas por ventas totales
        top_consoles = df.groupby('console')['total_sales'].sum().nlargest(6).index
        
        # Agrupar ventas por región para las top consolas
        region_sales = df[df['console'].isin(top_consoles)].groupby('console')[['na_sales','jp_sales','pal_sales','other_sales']].sum()
        
        bottom = [0] * len(region_sales)
        regiones = ['na_sales','jp_sales','pal_sales','other_sales']
        region_labels = ['Norte América', 'Japón', 'PAL', 'Otros']
        
        for i, (column, label) in enumerate(zip(regiones, region_labels)):
            ax.bar(region_sales.index, region_sales[column],
                  bottom=bottom, color=colors[i % len(colors)], 
                  label=label, alpha=0.8)
            bottom = [sum(x) for x in zip(bottom, region_sales[column])]
        
        ax.set_title("Ventas por región y plataforma", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Consola", color='white', fontsize=9)
        ax.set_ylabel("Ventas (millones)", color='white', fontsize=9)
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.legend(facecolor='#1a1a2e', edgecolor='#00ff88', fontsize=7)
        ax.grid(True, alpha=0.3)

def crear_grafico_top_publishers_consola(ax, df, colors):
    """Crea gráfico de top publishers por consola"""
    if "console" in df.columns and "publisher" in df.columns and "total_sales" in df.columns:
        # Top 5 consolas
        top_consoles = df['console'].value_counts().head(5).index
        # Top 5 publishers por ventas
        top_publishers = df.groupby('publisher')['total_sales'].sum().nlargest(5).index
        
        # Preparar datos para el heatmap
        console_publisher_data = []
        for console in top_consoles:
            console_df = df[df['console'] == console]
            publisher_sales = console_df.groupby('publisher')['total_sales'].sum()
            for publisher in top_publishers:
                console_publisher_data.append({
                    'console': console,
                    'publisher': publisher,
                    'sales': publisher_sales.get(publisher, 0)
                })
        
        heatmap_data = pd.DataFrame(console_publisher_data)
        pivot_data = heatmap_data.pivot(index='console', columns='publisher', values='sales').fillna(0)
        
        # Gráfico de barras apiladas
        bottom = [0] * len(pivot_data)
        for i, publisher in enumerate(pivot_data.columns):
            # Acortar nombres largos de publishers
            publisher_label = publisher[:15] + "..." if len(publisher) > 15 else publisher
            ax.bar(pivot_data.index, pivot_data[publisher], bottom=bottom,
                  color=colors[i % len(colors)], label=publisher_label, alpha=0.8)
            bottom = [sum(x) for x in zip(bottom, pivot_data[publisher])]
        
        ax.set_title("Top publishers por plataforma", color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel("Consola", color='white', fontsize=9)
        ax.set_ylabel("Ventas Totales", color='white', fontsize=9)
        ax.tick_params(axis='x', rotation=45, colors='white')
        ax.legend(facecolor='#1a1a2e', edgecolor='#00ff88', fontsize=6)
        ax.grid(True, alpha=0.3)



def crear_grafico_generos_exitosos_consola(ax, df, colors):
    """Crea gráfico de géneros más exitosos por consola"""
    if "console" in df.columns and "genre" in df.columns and "total_sales" in df.columns:
        # Top 5 consolas
        top_consoles = df['console'].value_counts().head(5).index
        
        # Encontrar el género más vendido por consola
        best_genres_by_console = df[df['console'].isin(top_consoles)]
        best_genres = best_genres_by_console.loc[best_genres_by_console.groupby('console')['total_sales'].idxmax()]
        
        # Ordenar por ventas
        best_genres = best_genres.sort_values('total_sales', ascending=False)
        
        bars = ax.bar(range(len(best_genres)), 
                     best_genres['total_sales'], 
                     color=colors[:len(best_genres)],
                     alpha=0.8)
        
        ax.set_xticks(range(len(best_genres)))
        # Crear labels informativos
        labels = [f"{row['console']}\n{row['genre']}" for _, row in best_genres.iterrows()]
        ax.set_xticklabels(labels, color='white', fontsize=8, rotation=45, ha='right')
        ax.set_title("Género más exitoso por consola", color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel("Ventas Totales (millones)", color='white', fontsize=9)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}M', ha='center', va='bottom',
                   color='white', fontweight='bold', fontsize=8)
        
        ax.grid(True, alpha=0.3)















##funcion principal
def create_console_analysis(df, parent):
    """Función principal que crea el panel completo de análisis de consolas"""
    # Configurar estilo de matplotlib
    plt.style.use('dark_background')
    
    # Crear frame scrolleable
    main_frame, scrollable_frame, canvas = crear_frame_scrolleable(parent)
    
    # Calcular ancho disponible
    scrollable_frame.update_idletasks()
    available_width = max(scrollable_frame.winfo_width() - 40, 1200)
    
    # Crear gráficas
    crear_primera_grafica_consolas(scrollable_frame, df, available_width)
    crear_segunda_grafica_consolas(scrollable_frame,df,available_width)
    return main_frame


def create_console_grid_visualizations(df, parent):
    """Función legacy - usar create_console_analysis en su lugar"""
    return create_console_analysis(df, parent)