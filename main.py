import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
import sys
import os

# Importa tus clases ya creadas
from Scripts.analizador_crecimiento_pib import AnalizadorCrecimientoPIB_Final
from Scripts.analizador_inflacion_pibreal import AnalizadorInflacionPIBReal, AnalizadorPIBCompleto


class InterfazAnalisisPIB:
    def __init__(self, root):
        self.root = root
        self.root.state('normal')
        self.root.title("Dashboard Económico Completo - Análisis del PIB e Inflación - República Dominicana")
        self.root.geometry("1200x700")
        self.root.minsize(1200, 700)
        self.root.configure(bg="#f4f6f8")

        # Instancias de los analizadores
        self.analizador_crecimiento = AnalizadorCrecimientoPIB_Final()
        self.analizador_completo = AnalizadorPIBCompleto()
        self.analizador_inflacion = AnalizadorInflacionPIBReal()

        # Variables para controlar el estado
        self.datos_cargados = False
        self.figura_actual = None
        self.canvas_actual = None
        self.df_tasa_crecimiento = None

        # Lista de datos del PIB Corriente para inyectar
        self.nuevos_datos_pib = [
            11471.00, 12987.60, 14368.80, 16088.70, 17516.80, 19822.30, 21230.40,
            22002.70, 24107.00, 25261.10, 25770.20, 20845.70, 23186.60, 35911.70,
            38059.10, 44093.70, 48187.80, 48278.10, 53833.90, 57997.10, 60624.30,
            62608.10, 67091.40, 71049.30, 75613.10, 79103.00, 84974.20, 89113.20,
            78481.40, 95122.80, 113908.10, 120759.60, 124597.80
        ]

        # === Estilo general ===
        self.configurar_estilos()

        # === REORGANIZACIÓN: Marco principal con estructura fija ===
        self.frame_principal = ttk.Frame(root)
        self.frame_principal.pack(fill="both", expand=True, padx=10, pady=10)

        # === BARRA SUPERIOR FIJA CON CONTROLES PERMANENTES ===
        self.frame_controles_superior = ttk.Frame(self.frame_principal)
        self.frame_controles_superior.pack(fill="x", pady=(0, 0))
        
        # Estado de la aplicación
        self.label_estado = ttk.Label(self.frame_controles_superior, text="Estado: Esperando carga de datos...",
                                     foreground="red", font=("Segoe UI", 9, "bold"))
        self.label_estado.pack(side="left", padx=(0, 20))
        
        # Controles permanentes
        frame_botones_permanentes = ttk.Frame(self.frame_controles_superior)
        frame_botones_permanentes.pack(side="right")
        
        ttk.Button(frame_botones_permanentes, text=" Guardar Gráfico",
                  command=self.guardar_grafico).pack(side="left", padx=5)
        ttk.Button(frame_botones_permanentes, text=" Limpiar Todo",
                  command=self.limpiar_todo).pack(side="left", padx=5)
        ttk.Button(frame_botones_permanentes, text=" Salir",
                  command=self.root.destroy).pack(side="left", padx=5)

        # === CONTENEDOR PRINCIPAL (sidebar + área visualización) ===
        self.frame_contenedor = ttk.Frame(self.frame_principal)
        self.frame_contenedor.pack(fill="both", expand=True)

        # === Panel izquierdo: controles y texto de salida ===
        frame_izq = ttk.Frame(self.frame_contenedor, width=300)
        frame_izq.pack(side="left", fill="both", expand=False, padx=(0, 10))
        frame_izq.pack_propagate(False)

        # Panel de controles (sin cambios)
        frame_controles = ttk.LabelFrame(frame_izq, text=" Controles de Análisis", padding=10)
        frame_controles.pack(fill="x", pady=(0, 10))

        # Sección de carga de datos
        ttk.Label(frame_controles, text="Carga de Datos:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(frame_controles, text=" Cargar Datos Automáticos",
                  command=self.cargar_datos, style="Accent.TButton").pack(fill="x", pady=2)
        ttk.Button(frame_controles, text=" Seleccionar CSV Manualmente",
                  command=self.cargar_csv_manual).pack(fill="x", pady=2)

        ttk.Separator(frame_controles, orient='horizontal').pack(fill="x", pady=5)

        # Sección de análisis de crecimiento
        ttk.Label(frame_controles, text="Análisis de Crecimiento PIB:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(frame_controles, text=" Ejecutar Análisis Completo",
                  command=self.analizar_crecimiento_completo).pack(fill="x", pady=2)
        ttk.Button(frame_controles, text=" Generar Reporte",
                  command=self.generar_reporte_crecimiento).pack(fill="x", pady=2)

        ttk.Separator(frame_controles, orient='horizontal').pack(fill="x", pady=5)

        # Sección de análisis de inflación
        ttk.Label(frame_controles, text="Análisis de Inflación:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(frame_controles, text=" Analizar Inflación y PIB Real",
                  command=self.analizar_inflacion).pack(fill="x", pady=2)
        ttk.Button(frame_controles, text=" Generar Reporte Inflación",
                  command=self.generar_reporte_inflacion).pack(fill="x", pady=2)

        ttk.Separator(frame_controles, orient='horizontal').pack(fill="x", pady=5)

        # Sección de análisis de tasa de crecimiento
        ttk.Label(frame_controles, text="Análisis de Tasa de Crecimiento:", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(frame_controles, text=" Consultar Tasa de Crecimiento",
                  command=self.consultar_tasa_crecimiento).pack(fill="x", pady=2)
        ttk.Button(frame_controles, text=" Evolución del PIB",
                  command=self.evolucion_pib).pack(fill="x", pady=2)
        ttk.Button(frame_controles, text=" Estadísticas Promedio",
                  command=self.estadisticas_promedio).pack(fill="x", pady=2)

        # Panel para entrada de años
        frame_periodo = ttk.Frame(frame_controles)
        frame_periodo.pack(fill="x", pady=5)

        ttk.Label(frame_periodo, text="Año inicio:").pack(side="left")
        self.entry_inicio = ttk.Entry(frame_periodo, width=8)
        self.entry_inicio.pack(side="left", padx=5)

        ttk.Label(frame_periodo, text="Año fin:").pack(side="left")
        self.entry_fin = ttk.Entry(frame_periodo, width=8)
        self.entry_fin.pack(side="left", padx=5)

        ttk.Button(frame_controles, text=" Graficar por Período",
                  command=self.graficar_por_periodo).pack(fill="x", pady=2)

        # Panel de salida de texto
        frame_salida = ttk.LabelFrame(frame_izq, text=" Salida de Análisis", padding=5)
        frame_salida.pack(fill="both", expand=True)

        self.text_output = ScrolledText(frame_salida, wrap="word", width=65, height=30,
                                       font=("Consolas", 9), bg="#2b2b2b", fg="#ffffff",
                                       insertbackground="white")
        self.text_output.pack(fill="both", expand=True)

        # === Panel derecho: visualizaciones (ÁREA QUE SE LIMPIARÁ) ===
        frame_der = ttk.Frame(self.frame_contenedor)
        frame_der.pack(side="right", fill="both", expand=True)

        # Panel de visualización (este es el que se limpia)
        frame_viz = ttk.LabelFrame(frame_der, text=" Visualización Gráfica", padding=5)
        frame_viz.pack(fill="both", expand=True)

        self.frame_grafico = ttk.Frame(frame_viz)
        self.frame_grafico.pack(fill="both", expand=True)

    def configurar_estilos(self):
        """Configura los estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores y estilos
        style.configure('TFrame', background='#f4f6f8')
        style.configure('TLabel', background='#f4f6f8', font=('Segoe UI', 9))
        style.configure('TButton', font=('Segoe UI', 9), padding=6)
        style.configure('Accent.TButton', background='#3498db', foreground='white')
        style.configure('TLabelframe', background='#f4f6f8', font=('Segoe UI', 9, 'bold'))
        style.configure('TLabelframe.Label', background='#f4f6f8', font=('Segoe UI', 9, 'bold'))

    def limpiar_todo(self):
        """Limpia el área de visualización pero mantiene los controles"""
        # Solo limpia el frame de gráficos, no los controles superiores
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()
        
        self.text_output.delete(1.0, tk.END)
        self.label_estado.config(text="Estado: Limpiado", foreground="orange")
        
        # Limpiar referencias
        self.figura_actual = None
        self.canvas_actual = None

    # ======================
    # FUNCIONES DE CARGA DE DATOS
    # ======================

    def cargar_datos(self):
        """Carga todos los datasets automáticamente"""
        try:
            self.mostrar_mensaje("Cargando datos de República Dominicana...")

            # Cargar datos para crecimiento
            if self.analizador_crecimiento.cargar_datos():
                self.datos_cargados = True
                self.label_estado.config(text="Estado: Datos cargados correctamente", foreground="green")
                self.mostrar_mensaje(" Todos los datasets cargados exitosamente para análisis de crecimiento")

                # Mostrar información sobre los datos cargados
                self.mostrar_info_datos_cargados()
            else:
                self.mostrar_mensaje(" Error al cargar datos de crecimiento")
                return

            messagebox.showinfo("Datos cargados",
                                "Todos los datos de República Dominicana fueron cargados correctamente.")

        except Exception as e:
            error_msg = f"Error al cargar datos:\n{str(e)}"
            self.mostrar_mensaje(f" {error_msg}")
            messagebox.showerror("Error al cargar", error_msg)

    def cargar_csv_manual(self):
        """Permite seleccionar archivos CSV manualmente"""
        archivos = filedialog.askopenfilenames(
            filetypes=[("CSV Files", "*.csv")],
            title="Selecciona uno o varios archivos CSV"
        )

        if archivos:
            try:
                dataframes = []
                for archivo in archivos:
                    temp_df = self.leer_csv_auto(archivo)
                    dataframes.append(temp_df)

                self.df_tasa_crecimiento = pd.concat(dataframes, ignore_index=True).drop_duplicates()
                self.df_tasa_crecimiento.reset_index(drop=True, inplace=True)

                # Actualizar datos del PIB Corriente
                self.actualizar_pib_corriente()

                messagebox.showinfo(
                    "Éxito",
                    f"Se cargaron {len(archivos)} archivos correctamente.\n"
                    f"Total de registros combinados: {len(self.df_tasa_crecimiento)}"
                )

                self.mostrar_mensaje(f" Se cargaron {len(archivos)} archivos CSV manualmente")
                self.datos_cargados = True
                self.label_estado.config(text="Estado: Datos CSV cargados correctamente", foreground="green")

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar los CSV:\n{e}")

    def leer_csv_auto(self, ruta_csv):
        """Lee CSV automáticamente detectando separador"""
        with open(ruta_csv, 'r', encoding='utf-8') as f:
            primero = f.readline()
            sep = ';' if primero.count(';') > primero.count(',') else ','
        try:
            df_temp = pd.read_csv(ruta_csv, sep=sep, decimal='.')
        except:
            df_temp = pd.read_csv(ruta_csv, sep=sep, decimal=',')
        return df_temp

    def actualizar_pib_corriente(self):
        """Actualiza la columna PIB Corriente con los datos predefinidos"""
        if self.df_tasa_crecimiento is None or self.df_tasa_crecimiento.empty:
            return False

        n_df = len(self.df_tasa_crecimiento)
        n_datos = len(self.nuevos_datos_pib)

        if n_df > n_datos:
            self.df_tasa_crecimiento['PIB Corriente.3'] = pd.Series(self.nuevos_datos_pib,
                                                                    index=self.df_tasa_crecimiento.index[:n_datos])
            self.mostrar_mensaje("⚠ DataFrame más largo que datos disponibles - se actualizaron primeras filas")
        else:
            self.df_tasa_crecimiento['PIB Corriente.3'] = np.resize(self.nuevos_datos_pib, n_df)

        self.mostrar_mensaje(" Datos de PIB Corriente actualizados")
        return True

    # ======================
    # NUEVAS FUNCIONALIDADES DE TASA DE CRECIMIENTO
    # ======================

    def consultar_tasa_crecimiento(self):
        """Muestra la gráfica de tasa de crecimiento"""
        if not self.verificar_datos_tasa_crecimiento():
            return

        self.limpiar_grafico()
        self.mostrar_mensaje("\n CONSULTANDO TASA DE CRECIMIENTO")
        self.mostrar_mensaje("=" * 50)

        # Crear frame para estadísticas
        stats_frame = ttk.Frame(self.frame_grafico)
        stats_frame.pack(fill="x", pady=10)

        # Calcular estadísticas
        promedio = self.df_tasa_crecimiento['PIB Referencia 2018'].mean()
        maximo = self.df_tasa_crecimiento['PIB Referencia 2018'].max()
        minimo = self.df_tasa_crecimiento['PIB Referencia 2018'].min()

        self.crear_tarjeta(stats_frame, "Crecimiento Promedio", f"{promedio:.2f}%")
        self.crear_tarjeta(stats_frame, "Variación Máx.", f"{maximo:.2f}%")
        self.crear_tarjeta(stats_frame, "Variación Mín.", f"{minimo:.2f}%")

        # Proyección tasa de crecimiento siguiente año
        try:
            if 'PIB Corriente.3' in self.df_tasa_crecimiento.columns and len(self.df_tasa_crecimiento) >= 2:
                pib_actual = self.df_tasa_crecimiento['PIB Corriente.3'].iloc[-2]
                pib_siguiente = self.df_tasa_crecimiento['PIB Corriente.3'].iloc[-1]
                tasa_siguiente = ((pib_siguiente / pib_actual) - 1) * 100
                self.crear_tarjeta(stats_frame, "Proyección Tasa Siguiente Año", f"{tasa_siguiente:.2f}%")
        except:
            self.crear_tarjeta(stats_frame, "Proyección Tasa Siguiente Año", "N/D")

        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_tasa_crecimiento['Período'], self.df_tasa_crecimiento['PIB Referencia 2018'],
                marker='o', color='#007acc', linewidth=2)
        ax.set_title('Tasa de Crecimiento del PIB - República Dominicana', fontsize=12, fontweight='bold')
        ax.set_xlabel('Año')
        ax.set_ylabel('Tasa de Crecimiento (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en los puntos
        for i, (x, y) in enumerate(
                zip(self.df_tasa_crecimiento['Período'], self.df_tasa_crecimiento['PIB Referencia 2018'])):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, fontweight='bold')

        fig.tight_layout()
        self.mostrar_grafico(fig)
        self.mostrar_mensaje(" Gráfica de tasa de crecimiento generada")

    def evolucion_pib(self):
        """Muestra la evolución del PIB Corriente"""
        if not self.verificar_datos_tasa_crecimiento():
            return

        if 'PIB Corriente.3' not in self.df_tasa_crecimiento.columns:
            messagebox.showwarning("Aviso", "La columna 'PIB Corriente.3' no está disponible.")
            return

        self.limpiar_grafico()
        self.mostrar_mensaje("\n EVOLUCIÓN DEL PIB CORRIENTE")
        self.mostrar_mensaje("=" * 50)

        # Crear frame para estadísticas
        stats_frame = ttk.Frame(self.frame_grafico)
        stats_frame.pack(fill="x", pady=10)

        # Calcular estadísticas
        promedio = self.df_tasa_crecimiento['PIB Corriente.3'].mean()
        maximo = self.df_tasa_crecimiento['PIB Corriente.3'].max()
        minimo = self.df_tasa_crecimiento['PIB Corriente.3'].min()

        self.crear_tarjeta(stats_frame, "Promedio PIB Corriente", f"{promedio:,.2f}")
        self.crear_tarjeta(stats_frame, "Máximo PIB Corriente", f"{maximo:,.2f}")
        self.crear_tarjeta(stats_frame, "Mínimo PIB Corriente", f"{minimo:,.2f}")

        # Predicción para siguiente año
        try:
            pib_inicial = self.df_tasa_crecimiento['PIB Corriente.3'].iloc[0]
            pib_final = self.df_tasa_crecimiento['PIB Corriente.3'].iloc[-1]
            tasa_crecimiento = (pib_final - pib_inicial) / pib_inicial
            next_year = self.df_tasa_crecimiento['Período'].max() + 1
            pib_siguiente = pib_final * (1 + tasa_crecimiento)
            self.crear_tarjeta(stats_frame, f"Predicción PIB {next_year}", f"{pib_siguiente:,.2f}")
        except:
            self.crear_tarjeta(stats_frame, "Predicción PIB", "N/D")

        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.df_tasa_crecimiento['Período'], self.df_tasa_crecimiento['PIB Corriente.3'],
                marker='o', color='#28a745', linewidth=2)
        ax.set_title('Evolución del PIB Corriente - República Dominicana', fontsize=12, fontweight='bold')
        ax.set_xlabel('Año')
        ax.set_ylabel('PIB Corriente')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Formatear eje Y con separadores de miles
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

        fig.tight_layout()
        self.mostrar_grafico(fig)
        self.mostrar_mensaje(" Gráfica de evolución del PIB generada")

    def graficar_por_periodo(self):
        """Grafica la tasa de interés por período específico"""
        I = self.entry_inicio.get()
        F = self.entry_fin.get()

        if not I.isdigit() or not F.isdigit():
            messagebox.showwarning("Aviso", "Debes ingresar años válidos.")
            return

        I = int(I)
        F = int(F)

        if not self.verificar_datos_tasa_crecimiento():
            return

        df_periodos = self.df_tasa_crecimiento[['Período', 'PIB Referencia 2018']].copy()
        rango = df_periodos[(df_periodos["Período"] >= I) & (df_periodos["Período"] <= F)]

        if rango.empty:
            messagebox.showwarning("Aviso", "No hay datos en ese rango.")
            return

        self.limpiar_grafico()
        self.mostrar_mensaje(f"\n TASA DE CRECIMIENTO {I}-{F}")
        self.mostrar_mensaje("=" * 50)

        # Calcular estadísticas del período
        crecimiento_promedio = rango['PIB Referencia 2018'].mean()
        maximo = rango['PIB Referencia 2018'].max()
        minimo = rango['PIB Referencia 2018'].min()

        # Crear frame para estadísticas
        stats_frame = ttk.Frame(self.frame_grafico)
        stats_frame.pack(fill="x", pady=10)

        self.crear_tarjeta(stats_frame, "Crecimiento Promedio", f"{crecimiento_promedio:.2f}%")
        self.crear_tarjeta(stats_frame, "Variación Máx.", f"{maximo:.2f}%")
        self.crear_tarjeta(stats_frame, "Variación Mín.", f"{minimo:.2f}%")

        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rango['Período'], rango['PIB Referencia 2018'],
                marker='o', color='#ff6b6b', linewidth=2)
        ax.set_title(f'Tasa de Crecimiento del PIB {I}-{F} - República Dominicana',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Año')
        ax.set_ylabel('Tasa de Crecimiento (%)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en los puntos
        for i, (x, y) in enumerate(zip(rango['Período'], rango['PIB Referencia 2018'])):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, fontweight='bold')

        fig.tight_layout()
        self.mostrar_grafico(fig)
        self.mostrar_mensaje(f" Gráfica del período {I}-{F} generada")

    def estadisticas_promedio(self):
        """Muestra estadísticas promedio del PIB"""
        if not self.verificar_datos_tasa_crecimiento():
            return

        promedio = self.df_tasa_crecimiento['PIB Referencia 2018'].mean()

        self.mostrar_mensaje("\n ESTADÍSTICAS PROMEDIO")
        self.mostrar_mensaje("=" * 40)
        self.mostrar_mensaje(f"El PIB promedio es: {promedio:.2f}%")

        messagebox.showinfo("Estadísticas promedio", f"El PIB promedio es: {promedio:.2f}%")

    def crear_tarjeta(self, parent, titulo, valor):
        """Crea una tarjeta de estadística"""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.pack(side="left", padx=5, pady=5, fill="x", expand=True)

        ttk.Label(card, text=titulo, font=("Segoe UI", 9, "bold"),
                  foreground="#1e3a5f", style='Card.TLabel').pack(pady=(5, 2))
        ttk.Label(card, text=valor, font=("Segoe UI", 10, "bold"),
                  foreground="#007acc", style='Card.TLabel').pack(pady=(2, 5))
        return card

    def verificar_datos_tasa_crecimiento(self):
        """Verifica si hay datos de tasa de crecimiento disponibles"""
        if self.df_tasa_crecimiento is None or self.df_tasa_crecimiento.empty:
            messagebox.showwarning("Aviso", "Primero debes cargar datos CSV manualmente.")
            return False
        return True



    def mostrar_info_datos_cargados(self):
        """Muestra información sobre los datos cargados"""
        try:
            self.mostrar_mensaje("\n INFORMACIÓN DE DATOS CARGADOS:")
            self.mostrar_mensaje("-" * 40)

            if self.analizador_crecimiento.datos_pib_gasto is not None:
                años_pib = self.obtener_años_disponibles(self.analizador_crecimiento.datos_pib_gasto)
                self.mostrar_mensaje(f"• PIB por gasto: {len(años_pib)} años ({min(años_pib)}-{max(años_pib)})")

            if self.analizador_crecimiento.datos_imae is not None:
                self.mostrar_mensaje(f"• IMAE: {len(self.analizador_crecimiento.datos_imae)} registros")

            if self.analizador_crecimiento.datos_tasa_crecimiento is not None:
                self.mostrar_mensaje(
                    f"• Tasas de crecimiento: {len(self.analizador_crecimiento.datos_tasa_crecimiento)} registros")

            if self.df_tasa_crecimiento is not None:
                self.mostrar_mensaje(f"• Datos CSV manuales: {len(self.df_tasa_crecimiento)} registros")

            self.mostrar_mensaje("-" * 40)

        except Exception as e:
            self.mostrar_mensaje(f"️ Error mostrando info de datos: {str(e)}")

    def obtener_años_disponibles(self, datos_pib):
        """Extrae los años disponibles del dataset de PIB"""
        try:
            años = []
            for col in datos_pib.columns:
                if col.isdigit() and len(col) == 4:
                    años.append(int(col))
                elif 'COMPONENTES' not in col and any(char.isdigit() for char in col):
                    for parte in str(col).split():
                        if parte.isdigit() and len(parte) == 4:
                            años.append(int(parte))
            return sorted(años) if años else [2018, 2019, 2020, 2021, 2022, 2023]
        except:
            return [2018, 2019, 2020, 2021, 2022, 2023]



    # ======================
    # ANALIZAR CRECIMIENTO DEL PIB
    # ======================

    def analizar_crecimiento_completo(self):
        """Ejecuta el análisis completo de crecimiento del PIB"""
        if not self.datos_cargados:
            messagebox.showwarning("Datos no cargados", "Por favor, cargue los datos primero.")
            return

        self._ejecutar_con_redireccion(
            self.analizador_crecimiento.generar_reporte_analisis,
            self.visualizar_crecimiento_adaptado,
            "Análisis de Crecimiento PIB - República Dominicana"
        )

    def visualizar_crecimiento_adaptado(self):
        try:
            fig = self.analizador_crecimiento.visualizar_analisis_completo()
            if fig is not None:
                return self._ajustar_figura(fig)
            return None
        except Exception as e:
            self.mostrar_mensaje(f" Error en visualización de crecimiento: {str(e)}")
            return None

    def analizar_inflacion(self):
        """Ejecuta el análisis de inflación y PIB real"""
        if not self.datos_cargados:
            messagebox.showwarning("Datos no cargados", "Por favor, cargue los datos primero.")
            return

        try:
            self.mostrar_mensaje("Iniciando análisis de inflación para República Dominicana...")
            self.cargar_datos_inflacion_estimados()

            datos_pib_nominal = self.analizador_inflacion.extraer_pib_nominal_desde_datos(
                self.analizador_crecimiento.datos_pib_gasto
            )

            if datos_pib_nominal is not None:
                self.analizador_inflacion.datos_pib_nominal = datos_pib_nominal
                self.mostrar_mensaje(" Datos de PIB nominal extraídos correctamente")

                pib_real = self.analizador_inflacion.calcular_pib_real(ano_base=2018)

                if pib_real is not None:
                    self._ejecutar_con_redireccion(
                        self.analizador_inflacion.generar_reporte_inflacion,
                        self.visualizar_inflacion_adaptado,
                        "Análisis de Inflación y PIB Real - República Dominicana"
                    )
                else:
                    messagebox.showerror("Error", "No se pudo calcular el PIB real")
            else:
                messagebox.showerror("Error", "No se pudieron extraer datos del PIB nominal")

        except Exception as e:
            error_msg = f"Error en análisis de inflación:\n{str(e)}"
            self.mostrar_mensaje(f" {error_msg}")
            messagebox.showerror("Error", error_msg)

    def visualizar_inflacion_adaptado(self):
        """Visualización adaptada para inflación"""
        try:
            fig = self.analizador_inflacion.visualizar_inflacion_pib_real()
            if fig is not None:
                return self._ajustar_figura(fig)
            return None
        except Exception as e:
            self.mostrar_mensaje(f" Error en visualización de inflación: {str(e)}")
            return None

    def _ajustar_figura(self, fig):
        """Ajusta una figura para que quepa mejor en la interfaz"""
        try:
            fig.set_size_inches(9, 7)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            for ax in fig.get_axes():
                ax.tick_params(axis='x', labelsize=8, rotation=45)
                ax.tick_params(axis='y', labelsize=8)

                title = ax.get_title()
                if title:
                    ax.set_title(title, fontsize=10, pad=5)

                xlabel = ax.get_xlabel()
                if xlabel:
                    ax.set_xlabel(xlabel, fontsize=9)

                ylabel = ax.get_ylabel()
                if ylabel:
                    ax.set_ylabel(ylabel, fontsize=9)

                legend = ax.get_legend()
                if legend:
                    legend.set_fontsize(8)

            return fig

        except Exception as e:
            self.mostrar_mensaje(f"⚠ Error ajustando figura: {str(e)}")
            return fig

    def cargar_datos_inflacion_estimados(self):
        """Carga datos de inflación estimados para RD"""
        try:
            inflacion_data_rd = [
                [2018, 3.5], [2019, 3.2], [2020, 1.8], [2021, 4.5],
                [2022, 7.2], [2023, 6.1], [2024, 3.8], [2025, 2.9]
            ]

            self.analizador_inflacion.datos_inflacion = pd.DataFrame(
                inflacion_data_rd, columns=['Ano', 'Inflacion_Anual_%']
            )
            self.mostrar_mensaje(" Datos de inflación estimados para República Dominicana cargados")

        except Exception as e:
            self.mostrar_mensaje(f" Error cargando datos de inflación: {str(e)}")

    def generar_reporte_crecimiento(self):
        """Genera solo el reporte de crecimiento sin gráficos"""
        if not self.datos_cargados:
            messagebox.showwarning("Datos no cargados", "Por favor, cargue los datos primero.")
            return

        self.limpiar_grafico()
        self._redirigir_salida(self.analizador_crecimiento.generar_reporte_analisis)

    def generar_reporte_inflacion(self):
        """Genera solo el reporte de inflación sin gráficos"""
        if not self.datos_cargados:
            messagebox.showwarning("Datos no cargados", "Por favor, cargue los datos primero.")
            return

        self.limpiar_grafico()
        try:
            self.cargar_datos_inflacion_estimados()
            datos_pib_nominal = self.analizador_inflacion.extraer_pib_nominal_desde_datos(
                self.analizador_crecimiento.datos_pib_gasto
            )
            if datos_pib_nominal is not None:
                self.analizador_inflacion.datos_pib_nominal = datos_pib_nominal
                self.analizador_inflacion.calcular_pib_real(ano_base=2018)
                self._redirigir_salida(self.analizador_inflacion.generar_reporte_inflacion)
            else:
                messagebox.showerror("Error", "No se pudieron extraer datos del PIB nominal")
        except Exception as e:
            self.mostrar_mensaje(f" Error generando reporte de inflación: {str(e)}")

    def _ejecutar_con_redireccion(self, funcion_texto, funcion_grafico, titulo):
        """Función auxiliar para ejecutar análisis con redirección de salida"""
        self.limpiar_grafico()
        self.mostrar_mensaje(f"\n EJECUTANDO: {titulo}")
        self.mostrar_mensaje("=" * 60)

        self._redirigir_salida(funcion_texto)

        try:
            fig = funcion_grafico()
            if fig is not None:
                if hasattr(fig, 'suptitle'):
                    fig.suptitle(titulo, fontsize=12, fontweight='bold', y=0.98)
                self.mostrar_grafico(fig)
                self.mostrar_mensaje(f" Gráfico de {titulo} generado correctamente")
            else:
                self.mostrar_mensaje("⚠ No se pudo generar el gráfico")
        except Exception as e:
            error_msg = f"Error generando gráfico: {str(e)}"
            self.mostrar_mensaje(f" {error_msg}")
            messagebox.showwarning("Visualización", error_msg)

    def _redirigir_salida(self, funcion):
        """Redirige la salida estándar al widget de texto"""
        buffer = io.StringIO()
        sys_stdout_original = sys.stdout
        sys.stdout = buffer

        try:
            funcion()
        except Exception as e:
            self.mostrar_mensaje(f" Error durante la ejecución: {str(e)}")
        finally:
            sys.stdout = sys_stdout_original

        texto_resultado = buffer.getvalue()
        self.mostrar_mensaje(texto_resultado)

    def mostrar_grafico(self, figura):
        """Muestra una figura matplotlib en el panel de gráficos"""
        self.limpiar_grafico()

        self.figura_actual = figura
        self.canvas_actual = FigureCanvasTkAgg(figura, master=self.frame_grafico)
        self.canvas_actual.draw()
        self.canvas_actual.get_tk_widget().pack(fill="both", expand=True)

    def mostrar_mensaje(self, mensaje):
        """Muestra un mensaje en el panel de salida"""
        self.text_output.insert(tk.END, mensaje + "\n")
        self.text_output.see(tk.END)
        self.root.update()

    def limpiar_todo(self):
        """Limpia tanto el texto como los gráficos"""
        self.text_output.delete(1.0, tk.END)
        self.limpiar_grafico()
        self.mostrar_mensaje(" Todo limpiado - Listo para nuevo análisis")
        self.label_estado.config(text="Estado: Esperando carga de datos...", foreground="red")

    def limpiar_grafico(self):
        """Limpia el panel de gráficos"""
        for widget in self.frame_grafico.winfo_children():
            widget.destroy()
        self.figura_actual = None
        self.canvas_actual = None

    def guardar_grafico(self):
        """Guarda el gráfico actual como imagen"""
        if self.figura_actual is None:
            messagebox.showwarning("Sin gráfico", "No hay ningún gráfico para guardar.")
            return

        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grafico_rd_{timestamp}.png"

            self.figura_actual.savefig(filename, dpi=150, bbox_inches='tight')
            self.mostrar_mensaje(f" Gráfico guardado como: {filename}")
            messagebox.showinfo("Guardado exitoso", f"Gráfico guardado como:\n{filename}")
        except Exception as e:
            error_msg = f"Error guardando gráfico: {str(e)}"
            self.mostrar_mensaje(f" {error_msg}")
            messagebox.showerror("Error al guardar", error_msg)



# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    # Configurar matplotlib para mejores visualizaciones
    plt.rcParams['figure.figsize'] = [10, 8]
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8

    root = tk.Tk()
    app = InterfazAnalisisPIB(root)

    # Mensaje de bienvenida
    app.mostrar_mensaje("=" * 60)
    app.mostrar_mensaje("   DASHBOARD ECONÓMICO COMPLETO - REPÚBLICA DOMINICANA")
    app.mostrar_mensaje("   ANÁLISIS DE PIB, INFLACIÓN Y TASAS DE CRECIMIENTO")
    app.mostrar_mensaje("=" * 60)
    app.mostrar_mensaje("")
    app.mostrar_mensaje("Instrucciones:")
    app.mostrar_mensaje("1. Use 'Cargar Datos Automáticos' para análisis predefinidos")
    app.mostrar_mensaje("2. Use 'Seleccionar CSV Manualmente' para datos personalizados")
    app.mostrar_mensaje("3. Seleccione el tipo de análisis a ejecutar")
    app.mostrar_mensaje("4. Revise los resultados en los paneles")
    app.mostrar_mensaje("")

    root.mainloop()