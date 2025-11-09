import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
import pandas as pd

def crear_tarjeta(contenedor, titulo, valor, color_principal, color_fondo):
    """Crea una tarjeta de métrica con diseño moderno"""
    card = tk.Frame(contenedor, bg=color_fondo, relief=tk.RAISED, bd=1,
                    width=190, height=80, highlightbackground=color_principal, highlightthickness=1)
    card.pack(side="left", expand=True, padx=3, pady=3, fill="both")
    card.pack_propagate(False)
    
    #Esto es para que queden los iconos y textos centralizados
    canvas_izquierdo = tk.Canvas(card, width=30, height=80, bg=color_fondo, highlightthickness=0, relief='flat')
    canvas_central = tk.Canvas(card, width=130, height=80, bg=color_fondo, highlightthickness=0, relief='flat')
    canvas_derecho = tk.Canvas(card, width=30, height=80, bg=color_fondo, highlightthickness=0, relief='flat')
    
    canvas_izquierdo.grid(row=0, column=0, sticky="nsew")
    canvas_central.grid(row=0, column=1, sticky="nsew")
    canvas_derecho.grid(row=0, column=2, sticky="nsew")
    
    card.columnconfigure(0, weight=1)
    card.columnconfigure(1, weight=1)
    card.columnconfigure(2, weight=1)
    
    # Centrar elementos en el canvas central
    centro_x_central = 130 // 2
    
    canvas_central.create_text(centro_x_central, 20, text=titulo, fill="#ffffff", 
                              font=("Arial", 8, "bold"), justify="center")
    canvas_central.create_text(centro_x_central, 45, text=str(valor), fill=color_principal, 
                              font=("Arial", 12, "bold"), justify="center")
    
    # Línea 
    line_length = 40
    line_start_x = centro_x_central - line_length // 2
    line_end_x = centro_x_central + line_length // 2
    canvas_central.create_line(line_start_x, 60, line_end_x, 60, fill=color_principal, width=1)
    
    canvas_central.create_text(centro_x_central, 70, text="Games Analytics", fill="#888888", 
                              font=("Arial", 6), justify="center")

def crear_metricas_cards(df, parent):
    """Crea el frame con las tarjetas de métricas"""
    frame_cards_container = tk.Frame(parent, bg="#0f0f23")
    frame_cards_container.pack(fill="x", pady=10, padx=10)
    
    frame_cards = tk.Frame(frame_cards_container, bg="#0f0f23")
    frame_cards.pack(fill="x", pady=5)

    # Calcular métricas
    total_registros = len(df)
    total_generos = df["genre"].nunique()
    total_plataformas = df["console"].nunique()
    total_juegos = df["title"].nunique()
    total_publishers = df["publisher"].nunique()

    # Crear tarjetas
    crear_tarjeta(frame_cards, "🎮 Total Registros", f"{total_registros:,}", "#00e5ff", "#0a0a1a")
    crear_tarjeta(frame_cards, "📊 Géneros Únicos", total_generos, "#9c27b0", "#0a0a1a")
    crear_tarjeta(frame_cards, "🖥️ Plataformas", total_plataformas, "#00ff88", "#0a0a1a")
    crear_tarjeta(frame_cards, "🎯 Juegos Únicos", f"{total_juegos:,}", "#ff6d00", "#0a0a1a")
    crear_tarjeta(frame_cards, "🏢 Publishers", f"{total_publishers:,}", "#ffeb3b", "#0a0a1a")
    
    return frame_cards_container

def crear_scrollable_frame(parent):
    #Crea el frame scrolleable para los gráficos
    frame_graficos_container = tk.Frame(parent, bg="#0f0f23")
    frame_graficos_container.pack(fill="both", expand=True, padx=10, pady=10)

    # Crea canvas scrolleable
    canvas_container = tk.Canvas(frame_graficos_container, bg="#0f0f23", highlightthickness=0)
    scrollbar = tk.Scrollbar(frame_graficos_container, orient="vertical", command=canvas_container.yview)
    scrollable_frame = tk.Frame(canvas_container, bg="#0f0f23")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas_container.configure(scrollregion=canvas_container.bbox("all"))
    )

    canvas_container.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas_container.configure(yscrollcommand=scrollbar.set)

    canvas_container.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    return scrollable_frame, canvas_container

def crear_grafico_top_generos(ax, df, colors):
   # Gráfico de top géneros
    if "genre" in df.columns:
        top_generos = df["genre"].value_counts().head(5)
        bars = ax.barh(range(len(top_generos)), top_generos.values, color=colors[:5])
        ax.set_yticks(range(len(top_generos)))
        ax.set_yticklabels(top_generos.index, color='white', fontsize=9)
        ax.set_title(" ☆ Top 5 Géneros", color='white', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Cantidad", color='white', fontsize=9)
        ax.invert_yaxis()
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, str(int(width)), 
                   ha='left', va='center', color='white', fontweight='bold', fontsize=8)

def crear_grafico_top_plataformas(ax, df, colors):
      #Gráfico de top plataformas"""
    if "console" in df.columns:
        top_plataformas = df["console"].value_counts().head(6)
        total = top_plataformas.sum()
        porcentajes = (top_plataformas / total) * 100

        # Gráfico de barras horizontales
        barras = ax.barh(top_plataformas.index[::-1], porcentajes[::-1], color=colors[:6], alpha=0.8)

        # Mostrar porcentaje al final de cada barra
        for i, pct in enumerate(porcentajes[::-1]):
            ax.text(pct + 1, i, f"{pct:.1f}%", va='center', color='white', fontsize=8, fontweight='bold')

        # Estilos
        ax.set_title("Presencia de Plataformas", color='white', fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel("Porcentaje (%)", color='white', fontsize=9)
        ax.set_xlim(0, max(porcentajes)*1.2)
        ax.grid(True, alpha=0.2, axis='x')

def crear_grafico_regiones(ax, colors):
    #Gráfico de regiones
    regiones = ["PAL", "NA", "JP", "OTHERS"]
    wedges, texts = ax.pie([1,1,1,1], labels=regiones, colors=colors[:4],
                          startangle=45, textprops={'color':'white','fontsize':8,'fontweight':'bold'})
    ax.set_title("◕ Regiones de la Muestra", color='white', fontsize=11, fontweight='bold', pad=10)
    ax.axis('equal')

def crear_grafico_frecuencia_anual(ax, df, colors):
    #Gráfico de frecuencia por año
    if 'year' in df.columns:
        frecuencia_anual = df['year'].value_counts().sort_index()
        for i, (year, count) in enumerate(frecuencia_anual.items()):
            ax.bar(year, count, color=colors[i % len(colors)], edgecolor='black')
        
        ax.set_title("Presencia de Juegos en la muestra por año de lanzamiento", 
                    color='white', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Año de Lanzamiento", color='white', fontsize=10)
        ax.set_ylabel("Cantidad de Juegos", color='white', fontsize=10)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='white')

def aplicar_estilo_graficos(axes):
    #Aplica el estilo repetido a los graficos (fondos y colores)
    for ax in axes:
        ax.set_facecolor('#0a0a1a')
        for spine in ax.spines.values():
            spine.set_color('#00e5ff')
            spine.set_linewidth(1)
        ax.tick_params(colors='white', labelsize=8)

def crear_graficos_principales(df, scrollable_frame):
   #Crea y configura todos los gráficos principales
   
    plt.style.use('dark_background')
    
    # Crear figura y grid de las graficas
    fig = plt.figure(figsize=(12, 8), facecolor='#0f0f23')
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2], hspace=0.4, wspace=0.3)
    
    colors = ["#05bed3", '#9c27b0', "#07dd79", '#ff6d00', '#ffeb3b', '#e91e63', '#8e44ad', '#3498db']
    
    # Crear subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Top géneros
    ax2 = fig.add_subplot(gs[0, 1])  # Top plataformas
    ax3 = fig.add_subplot(gs[0, 2])  # Regiones
    ax4 = fig.add_subplot(gs[1, :])  # Frecuencia anual
    
    # Crear cada gráfico
    crear_grafico_top_generos(ax1, df, colors)
    crear_grafico_top_plataformas(ax2, df, colors)
    crear_grafico_regiones(ax3, colors)
    crear_grafico_frecuencia_anual(ax4, df, colors)
    
    # Aplicar estilo
    aplicar_estilo_graficos([ax1, ax2, ax3, ax4])
    
    # Ajustar layout
    fig.tight_layout(pad=2.0)
    
    # Integrar en Tkinter del padre
    canvas = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    # Añadir toolbar
    toolbar_frame = tk.Frame(scrollable_frame, bg="#0f0f23")
    toolbar_frame.pack(fill="x", padx=5, pady=2)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    toolbar.configure(background='#0f0f23')
    
    # Estilizar botones del toolbar
    for button in toolbar.winfo_children():
        if isinstance(button, tk.Button):
            button.configure(bg='#0a0a1a', fg='white', relief='flat')
    
    return canvas

def create_basic_analysis(df, parent):
    #Función principal que orquesta la creación del análisis completo
    # Preparar datos
    df_registros = df.copy()
    df_registros['release_date'] = pd.to_datetime(df_registros['release_date'], errors='coerce')
    df_registros['year'] = df_registros['release_date'].dt.year
    
    # Crear componentes en orden
    crear_metricas_cards(df_registros, parent)
    scrollable_frame, canvas_container = crear_scrollable_frame(parent)
    crear_graficos_principales(df_registros, scrollable_frame)