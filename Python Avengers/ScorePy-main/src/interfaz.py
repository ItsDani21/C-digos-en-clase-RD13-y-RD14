import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
from tkinter import filedialog, messagebox
import pandas as pd
from src.procesamiento import limpiar_datos
from src.generar_html import generar_reporte_html
from src.analisis_datos import (
    analisis_promedio_por_metodo,
    analisis_distractores,
    analisis_promedio_por_rango_horas,
    analisis_motivacion_vs_promedio,
    analisis_recursos_estudio_popularidad,
    analisis_promedio_por_frecuencia_repaso,
    analisis_sentimientos_vs_promedio,
)


# Carpeta de imágenes compatible con PyInstaller
if hasattr(sys, '_MEIPASS'):
    ASSETS_DIR = os.path.join(sys._MEIPASS, "img")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "img")


def load_image(name: str, size: tuple[int, int]):
    """Helper para cargar y redimensionar imágenes desde carpeta assets."""
    path = os.path.join(ASSETS_DIR, name)
    img = Image.open(path).resize(size)
    return ImageTk.PhotoImage(img)


class ExportedApp(ctk.CTk):
    """Ventana principal con lista de reportes dinámica y scroll.
    """

    def __init__(self):
        super().__init__()
        self.title("ScorePy")
        self.geometry("580x480")
        self.minsize(580, 480)
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'python.ico')
        else:
            icon_path = os.path.abspath('python.ico')
        self.iconbitmap(icon_path)

        # Reportes Disponibles
        self.report_names = [
            "Relacion Horas Estudio y Calificacion",   # horas
            "Relacion Metodo Estudio y Calificacion",  # metodos
            "Promedio por Metodo (tabla)",             # promedio_metodo
            "Promedio por Rango de Horas (tabla)",     # rango_horas
            "Impacto de Distractores",                 # distractores
            "Motivacion vs Promedio",                  # motivacion
            "Recursos de Estudio (frecuencia)",        # recursos
            "Frecuencia de Repaso vs Promedio",        # repaso
            "Sentimientos vs Promedio",                # sentimientos
        ]

        self.report_key_map = {
            "Relacion Horas Estudio y Calificacion": "horas",
            "Relacion Metodo Estudio y Calificacion": "metodos",
            "Promedio por Metodo (tabla)": "promedio_metodo",
            "Promedio por Rango de Horas (tabla)": "rango_horas",
            "Impacto de Distractores": "distractores",
            "Motivacion vs Promedio": "motivacion",
            "Recursos de Estudio (frecuencia)": "recursos",
            "Frecuencia de Repaso vs Promedio": "repaso",
            "Sentimientos vs Promedio": "sentimientos",
        }
        self.report_checkboxes = []

        self._dataframe = None

        self._build_static_ui()
        self._build_reports_panel()

    
    """
    < ------------------ Entrys ------------------ >
    """
    def _select_csv_file(self):
        """Abre un diálogo para seleccionar solo archivos CSV y coloca la ruta en el entry correspondiente."""
        path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV", filetypes=[("Archivos CSV", "*.csv")]
        )
        if path:
            self.input_csv.delete(0, "end")
            self.input_csv.insert(0, path)

    def _select_dest_dir(self):
        """Abre un diálogo para seleccionar carpeta de destino y coloca la ruta en el entry correspondiente."""
        path = filedialog.askdirectory(title="Seleccionar carpeta de destino")
        if path:
            self.input_destino.delete(0, "end")
            self.input_destino.insert(0, path)

    
    """
    < ------------------ Interfaz ------------------ >
    """
    def _build_static_ui(self):
        self.header_img = load_image("header.png", (570, 70))
        ctk.CTkLabel(self, width=570, height=70, image=self.header_img, text="").place(
            x=5, y=5
        )

        self.footer_img = load_image("footer.png", (570, 60))
        ctk.CTkLabel(self, width=570, height=60, image=self.footer_img, text="").place(
            x=5, y=415
        )

        self.left_img = load_image("archivos.png", (280, 300))
        ctk.CTkLabel(self, width=280, height=300, image=self.left_img, text="").place(
            x=5, y=95.5
        )

        self.right_img = load_image("reportes.png", (280, 300))
        ctk.CTkLabel(self, width=280, height=300, image=self.right_img, text="").place(
            x=295, y=95
        )

        entry_style = dict(
            fg_color="#FFFFFF", text_color="#000000", border_width=0, bg_color="#BEBEBE"
        )
        self.input_csv = ctk.CTkEntry(
            self,
            width=210,
            height=32,
            placeholder_text="Seleccione archivo...",
            corner_radius=7,
            **entry_style
        )
        self.input_csv.place(x=15, y=190)


        self.input_destino = ctk.CTkEntry(
            self,
            width=210,
            height=32,
            placeholder_text="Seleccione carpeta...",
            corner_radius=8,
            **entry_style
        )
        self.input_destino.place(x=15, y=308)

        # Nuevo: campo para nombre de archivo HTML
        self.input_nombre_html = ctk.CTkEntry(
            self,
            width=210,
            height=32,
            placeholder_text="Nombre del HTML (opcional)",
            corner_radius=8,
            **entry_style
        )
        self.input_nombre_html.place(x=15, y=350)


        btn_style = dict(
            fg_color="#FFFFFF",
            hover_color="#eeeeee",
            text_color="#FFFFFF",
            bg_color="#BEBEBE",
            border_width=0,
        )

        self.csv_icon = load_image("svg.png", (25, 25))
        self.btn_csv = ctk.CTkButton(
            self,
            text="",
            image=self.csv_icon,
            width=40,
            height=32,
            corner_radius=8,
            command=self._select_csv_file,
            **btn_style
        )
        self.btn_csv.place(x=236, y=190)

        self.destino_icon = load_image("destino.png", (25, 25))
        self.btn_destino = ctk.CTkButton(
            self,
            text="",
            image=self.destino_icon,
            width=40,
            height=32,
            corner_radius=8,
            command=self._select_dest_dir,
            **btn_style
        )
        self.btn_destino.place(x=236, y=308)

        self.btn_generar = ctk.CTkButton(
            self,
            text="Generar Reporte",
            width=190,
            height=36,
            corner_radius=6,
            fg_color="#155DFC",
            hover_color="#1f6aa5",
            text_color="#FFFFFF",
            bg_color="#BEBEBE",
            border_width=0,
            command=self._generate_report,
        )
        self.btn_generar.place(x=195, y=427)


    """
    < ------------------ Frame Reportes ------------------ >
    """
    def _build_reports_panel(self):
        self.reports_frame = ctk.CTkScrollableFrame(
            self,
            width=258,
            height=250,
            fg_color="#BEBEBE",
            border_width=0,
            corner_radius=0,
        )
        self.reports_frame.place(x=300, y=140)


        for name in self.report_names:
            self._add_report_checkbox(name)

    def _add_report_checkbox(self, name: str):
        """Crea un checkbox de reporte dentro del frame scrollable."""
        var = ctk.BooleanVar(value=False)
        cb = ctk.CTkCheckBox(
            self.reports_frame,
            text=name,
            width=250,
            height=30,
            fg_color="#155dff",
            hover_color="#0d4abf",
            text_color="#000000",
            border_color="#155dff",
            bg_color="transparent",
            variable=var,
            onvalue=True,
            offvalue=False,
        )
        cb.configure(fg_color="#155dff")
        cb.pack(anchor="w", pady=(5, 0))
        self.report_checkboxes.append((cb, var, name))
        return cb
    
    
    """
    < ------------------ Verificaciones ------------------ >
    """
    def _get_selected_report_keys(self):
        claves = []
        for _, var, nombre in self.report_checkboxes:
            if var.get():
                clave = self.report_key_map.get(nombre)
                if clave:
                    claves.append(clave)
        return claves

    def _load_and_clean_csv(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer el CSV:\n{e}")
            return None
        try:
            df_limpio = limpiar_datos(df)
            return df_limpio
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error limpiando los datos:\n{e}")
            return None

    def _generate_report(self):
        """Handler del botón 'Generar Reporte'."""
        csv_path = self.input_csv.get().strip()
        dest_dir = self.input_destino.get().strip()
        seleccionados = self._get_selected_report_keys()
        nombre_html = self.input_nombre_html.get().strip()


        if not csv_path:
            messagebox.showwarning("Falta CSV", "Seleccione un archivo CSV.")
            return
        if not os.path.isfile(csv_path):
            messagebox.showerror("Archivo inválido", "La ruta del CSV no es válida.")
            return
        if not dest_dir:
            messagebox.showwarning("Falta carpeta", "Seleccione una carpeta de destino.")
            return
        if not os.path.isdir(dest_dir):
            messagebox.showerror("Carpeta inválida", "La carpeta de destino no existe.")
            return
        if not seleccionados:
            messagebox.showinfo("Sin reportes", "Seleccione al menos un reporte de la lista.")
            return

        # Cargar y limpiar datos
        self.btn_generar.configure(state="disabled", text="Procesando...")
        self.update_idletasks()
        df = self._load_and_clean_csv(csv_path)
        if df is None:
            self.btn_generar.configure(state="normal", text="Generar Reporte")
            return
        self._dataframe = df

        # Construir nombre de archivo
        if nombre_html:
            output_path = os.path.join(dest_dir, nombre_html if nombre_html.lower().endswith('.html') else nombre_html + '.html')
        else:
            base_name = "reporte"
            try:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception:
                timestamp = "temp"
            output_path = os.path.join(dest_dir, f"{base_name}_{timestamp}.html")

        # Generar HTML
        try:
            generar_reporte_html(df, output_path, seleccionados, nombre_html if nombre_html else None)
        except Exception as e:
            messagebox.showerror("Error", f"Error generando el HTML:\n{e}")
            self.btn_generar.configure(state="normal", text="Generar Reporte")
            return

        self.btn_generar.configure(state="normal", text="Generar Reporte")
        messagebox.showinfo("Éxito", f"Reporte HTML generado:\n{output_path}")


if __name__ == "__main__":
    app = ExportedApp()
    app.mainloop()
