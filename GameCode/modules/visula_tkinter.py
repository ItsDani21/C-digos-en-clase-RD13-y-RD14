import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
from modules.data_loader import load_data
from modules.view_grafic import open_graph_view


#Esto es para buscar la carpeta de las imagenes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "../assets")

class App:
    def __init__(self, root):
        self.root = root
        root.title("GameSoft")
        root.configure(bg="#1e1e2f")
        root.geometry("1000x600")
        root.minsize(800, 500)

        # ---------------- NAVBAR ----------------
        nav = tk.Frame(root, bg="#ffffff", height=64)
        nav.pack(fill="x", side="top")

        # Logo a la izquierda
        logo_path = os.path.join(ASSETS_DIR, "controller.png")
        if os.path.exists(logo_path):
            logo_img = Image.open(logo_path).resize((40, 40))
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            tk.Label(nav, image=self.logo_photo, bg="#ffffff").pack(side="left", padx=15, pady=10)
        else:
            tk.Label(nav, text="GameSoft", bg="#ffffff", font=("Arial", 14, "bold")).pack(side="left", padx=15)

        tk.Label(nav, text="GameSoft", bg="#ffffff", font=("Helvetica", 14, "bold")).pack(side="left")

        

        #-----------Menu-----------------
        nav_links = tk.Frame(nav, bg="#ffffff")
        nav_links.pack(side="right", padx=20)
    
        for txt in ["Inicio", "Graficas", "Sobre Nosotros", "Informacion Tecnica"]:
            if txt == "Inicio":
                command = self.show_home
            elif txt == "Graficas":
                command = self.open_graphs
            elif txt == "Sobre Nosotros":
                command = self.show_about
            elif txt == "Informacion Tecnica":
                command = self.show_technical_info
            else:
                command = None
            
            b = tk.Button(nav_links, text=txt, bd=0, bg="#ffffff", 
                         activebackground="#f0f0ff", font=("Helvetica", 10),
                         command=command)  # Asignar comando aquí
            b.pack(side="left", padx=8, pady=14)

        # ---------------- HERO ----------------
        self.hero = tk.Frame(root, bg="#1e1e2f")
        self.hero.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Mostrar la página de inicio por defecto
        self.show_home()

        # ---------------- DATOS ---------------- #
        try:
            self.df = load_data()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudieron cargar los datos:\n{e}")
            self.df = None

    def clear_content(self):
        """Limpia el contenido actual"""
        for widget in self.hero.winfo_children():
            widget.destroy()

    def show_home(self):
        """Muestra la página de inicio pero solo cuando se abreeee"""
        self.clear_content()
        
        # Dos columnas esto es como css grid que tiene dos colummnas no tocar
        left = tk.Frame(self.hero, bg="#1e1e2f")
        left.pack(side="left", expand=True, fill="both", padx=20)

        right = tk.Frame(self.hero, bg="#1e1e2f")
        right.pack(side="right", expand=True, fill="both", padx=20)

        # Izquierda
        tk.Label(left, text="¡BIENVENIDOS A GAMESOFT!", font=("Arial Black", 28),
                 fg="white", bg="#1e1e2f", anchor="w").pack(anchor="w", pady=(50, 10))
        
        tk.Label(left, text="Transformando el Futuro del Gaming", 
                 font=("Helvetica", 16, "bold"), fg="#ff77ff", bg="#1e1e2f", anchor="w").pack(anchor="w", pady=(0, 20))

        tk.Label(left, text="""En GameSoft, combinamos innovación tecnológica con creatividad 
ilimitada para analizar datos y mediante criticas y analisis 

• Más de 15 gráficas
• Graficas Especializadas

Nuestro equipo de analistas y diseñadores trabaja incansablemente para generar estadísticas  visuales atractivas que reflejen \n las preferencias  en el gaming.
""",
                 font=("Helvetica", 12), fg="#cccccc", bg="#1e1e2f", 
                 justify="left", anchor="w").pack(anchor="w", pady=10)

        ttk.Button(left, text="Explorar Gráficas y Estadísticas", 
                   command=self.open_graphs).pack(anchor="w", pady=20)

        # Derecha: la imagen
        ctrl_path = os.path.join(ASSETS_DIR, "controller.png")
        if os.path.exists(ctrl_path):
            ctrl = Image.open(ctrl_path).resize((300, 300), Image.Resampling.LANCZOS)
            self.ctrl_photo = ImageTk.PhotoImage(ctrl)
            tk.Label(right, image=self.ctrl_photo, bg="#1e1e2f").pack(pady=20)
        else:
            tk.Label(right, text="🎮", font=("Arial", 120), 
                     fg="#ff77ff", bg="#1e1e2f").pack(pady=20)

        tk.Label(right, text="GAMESOFT", font=("Impact", 24),
                 fg="#ff77ff", bg="#1e1e2f").pack()

    def open_graphs(self):
        """Abre las gráficas"""
        if self.df is not None:
            open_graph_view(self.root, self.df)
        else:
            messagebox.showwarning("Aviso", "Los datos no están cargados")

    # Sobre nosotros 
    def show_about(self):
        self.clear_content()
        
        content = tk.Frame(self.hero, bg="#1e1e2f")
        content.pack(expand=True, fill="both", pady=50)
        
        tk.Label(content, text="SOBRE NOSOTROS", font=("Arial Black", 28),
                 fg="white", bg="#1e1e2f").pack(pady=20)
        
        tk.Label(content, text="""Somos C1nc0-b1ts, una empresa dedicada al analisis 
de la industria de los videojuegos.

GameSoft es una herramienta de software desarrollada por nuestro equipo que se encarga
de analizar datos de Metacritic sobre la industria de los videojuegos. 
Los resultados se muestran mediante gráficas 
generadas a partir de információn anual.

Nuestra filosofía se basa en tres pilares fundamentales:

📊 Innovación: Buscamos constantemente nuevas formas de representar los datos con creatividad y precisión.
🎨 Calidad: Cada gráfica refleja nuestro compromiso con la claridad, la estética y la exactitud.
🤝 Colaboración: Escuchamos a cada empresa para entender sus necesidades y transformar sus datos en información visualmente poderosa.
                 El nombre nuestro equipo es : C1nc0 b1ts"""
                ,
                 font=("Helvetica", 12), fg="#cccccc", bg="#1e1e2f",
                 justify="center").pack(pady=10)
        
        # Equipo directivo
        team_frame = tk.Frame(content, bg="#1e1e2f")
        team_frame.pack(pady=20)
        
        tk.Label(team_frame, text="C1nc0 B1ts:", font=("Helvetica", 14, "bold"),
                 fg="#ff77ff", bg="#1e1e2f").pack(pady=10)
        
        team_info = """
• Alvaro miguel
• Johaly Concepción
• Jeremy Hernández
• Manuel Rosario
"""
        tk.Label(team_frame, text=team_info, font=("Helvetica", 11),
                 fg="#aaaaaa", bg="#1e1e2f", justify="left").pack()
        

    # informacion Tecnica 
    def show_technical_info(self):
        """Muestra la página de Información Técnica"""
        self.clear_content()
        
        content = tk.Frame(self.hero, bg="#1e1e2f")
        content.pack(expand=True, fill="both", pady=50)
        
        tk.Label(content, text="INFORMACIÓN TÉCNICA", font=("Arial Black", 28),
                 fg="white", bg="#1e1e2f").pack(pady=20)
        
        # Información sobre los datos
        if self.df is not None:
            info_text = f"""
📊 DATOS CARGADOS CORRECTAMENTE:

• Filas totales: {len(self.df):,}
• Columnas disponibles: {len(self.df.columns)}
• Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB

📈 COLUMNAS DISPONIBLES PARA ANÁLISIS:
{', '.join(self.df.columns)}

🔍 ESTADÍSTICAS RÁPIDAS:
• Tipos de datos: {dict(self.df.dtypes)}
• Valores nulos: {self.df.isnull().sum().sum()}
• Valores duplicados: {self.df.duplicated().sum()}

📖LIBRERIAS UTILIZADAS:
• Pandas
• Matplotlib
• Seaborn
• Pillow
• Tkinter



"""
        else:
            info_text = "❌ NO HAY DATOS CARGADOS\n\nPor favor, verifica que el archivo de datos esté disponible y en el formato correcto."
        
        # Usar un widget Text para mejor formato
        text_widget = tk.Text(content, font=("Courier", 11), 
                             fg="#cccccc", bg="#1e1e2f", 
                             width=80, height=20, wrap=tk.WORD,
                             relief=tk.FLAT, borderwidth=0)
        text_widget.pack(pady=10, padx=20)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)  # Hacerlo de solo lectura
        
        # Scrollbar para el texto
        scrollbar = tk.Scrollbar(content, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)