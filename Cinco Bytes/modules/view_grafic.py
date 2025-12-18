import tkinter as tk
from tkinter import ttk
from modules.grafic import create_visualizations

def open_graph_view(root, df):
    # Crear ventana principal 
    win = tk.Toplevel(root)
    win.title("Visualizaciones - Dashboard")
    win.config(bg="#0f0f23")

    # Hacer que la ventana ocupe toda la pantalla del dispositivo
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    win.geometry(f"{screen_width}x{screen_height}")
    win.minsize(1000, 700)

    # Estilos 
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("Custom.TFrame", background="#0f0f23")
    style.configure("Card.TFrame", background="#1a1a2e", relief="flat", borderwidth=1)

    # Contenedor principal 
    main_container = tk.Frame(win, bg="#0f0f23")
    main_container.pack(fill="both", expand=True, padx=20, pady=20)

    content_container = tk.Frame(main_container, bg="#0f0f23")
    content_container.pack(fill="both", expand=True)

    #Menú lateral 
    nav_width = 270
    nav_frame = ttk.Frame(content_container, style="Card.TFrame", width=nav_width)
    nav_frame.pack(side="left", fill="y", padx=(0, 20))
    nav_frame.pack_propagate(False)

    tk.Label(
        nav_frame,
        text="CATEGORÍAS",
        font=("Segoe UI", 12, "bold"),
        fg="#00e5ff",
        bg="#1a1a2e"
    ).pack(pady=(20, 15))

    # Contenedor de gráficos 
    graph_container = ttk.Frame(content_container, style="Custom.TFrame")
    graph_container.pack(side="right", fill="both", expand=True)

    graph_main_frame = ttk.Frame(graph_container, style="Custom.TFrame")
    graph_main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    #Diccionario para guardar las gráficas 
    graph_frames = {}

    graphs = [
        {"title": "Σ Estadísticas básicas", "key": "generic"},
        {"title": "🎮 Análisis por Género", "key": "genero"},
        {"title": "🖳 Análisis por plataforma", "key": "plataformas"},
        {"title": "👥 Preferencias de usuarios", "key": "preferencias"},
         {"title": "💡 Insight", "key": "inversion"}

    ]

    # Crear frames de gráficas ---
    for g in graphs:
        # Frame contenedor para cada gráfica
        frame_container = ttk.Frame(graph_main_frame, style="Card.TFrame")
        frame_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame interno para el contenido
        frame = ttk.Frame(frame_container, style="Card.TFrame")
        frame.pack(fill="both", expand=True, padx=15, pady=15)

        tk.Label(
            frame,
            text=g["title"],
            font=("Segoe UI", 16, "bold"),  # Título más pequeño
            fg="#00e5ff",
            bg="#1a1a2e"
        ).pack(pady=(10, 15))

        # Frame específico para la visualización
        viz_frame = ttk.Frame(frame, style="Card.TFrame")
        viz_frame.pack(fill="both", expand=True, padx=5, pady=5)

        try:
            # Pasar el frame específico para las visualizaciones
         create_visualizations(df, g["title"]
                                  .replace("📈 ", "")
                                  .replace("👥 ", "")
                                  .replace("Σ ","")
                                  .replace("💡 ", "")
                                  .replace("🖳 ","")
                                  .replace("🎮 ", "")
                                  ,  
                                  viz_frame)

        except Exception as e:
            tk.Label(frame, text=f"Error: {str(e)}", fg="red", bg="#1a1a2e").pack(pady=20)

        graph_frames[g["key"]] = {
            "container": frame_container,
            "content": frame,
            "viz": viz_frame
        }
        
        # Ocultar inicialmente todos los frames
        frame_container.pack_forget()

    # Función para mostrar la gráfica seleccionada ---
    def mostrar_grafica(key):
        # Ocultar todos los frames primero
        for graph_data in graph_frames.values():
            graph_data["container"].pack_forget()
        
        # Mostrar el frame seleccionado
        selected_frame = graph_frames[key]["container"]
        selected_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Ajustar el tamaño máximo del frame contenedor
        available_width = screen_width - (nav_width + 80)  # 80px de padding/márgenes
        available_height = screen_height - 120  # 120px para header/footer
        
        # Configurar tamaño máximo del contenedor
        selected_frame.config(width=available_width, height=available_height)
        selected_frame.pack_propagate(False)  # Importante: evitar que se redimensione
        
        # Forzar actualización de la ventana
        win.update_idletasks()

    # Crear botones de navegación 
    def create_nav_button(parent, graph):
        btn = tk.Button(
            parent,
            text=graph["title"],
            bg="#252540",
            fg="#d6dede",
            relief="flat",
            font=("Segoe UI", 12, "bold"),
            activebackground="#1a1a2e",
            height=3,
            anchor="w",
            command=lambda k=graph["key"]: mostrar_grafica(k)
        )
        btn.pack(fill="x", padx=15, pady=8)

    for g in graphs:
        create_nav_button(nav_frame, g)

 
    # Mostrar la primera gráfica por defecto 
    if graphs:
        mostrar_grafica(graphs[0]["key"])

    # Función para ajustar tamaños cuando cambia la ventana 
    def on_resize(event=None):
        # Recalcular tamaños disponibles
        available_width = win.winfo_width() - (nav_width + 80)
        available_height = win.winfo_height() - 120
        
        # Aplicar a todos los frames visibles
        for graph_data in graph_frames.values():
            if graph_data["container"].winfo_ismapped():  # Si está visible
                graph_data["container"].config(
                    width=available_width, 
                    height=available_height
                )

    # Vincular el evento de redimensionamiento
    win.bind("<Configure>", on_resize)