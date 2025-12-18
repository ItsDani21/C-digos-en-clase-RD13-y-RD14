#fuciones 
from grafic_fun.genre_analysis import create_genre_analysis
from grafic_fun.basic_analysis import create_basic_analysis
from grafic_fun.console_analysis import create_console_analysis
from grafic_fun.user_preferences_analysis import create_user_analysis
from grafic_fun.insight_analysis import create_insight_analysis
def create_visualizations(df, option, parent):
    """Dirige a la visualización correspondiente"""
    if option == "Análisis por plataforma":
        return create_console_analysis(df, parent)
    elif option == "Análisis por Género":  
        return create_genre_analysis(df, parent)
    elif option == "Estadísticas básicas":
     return create_basic_analysis(df, parent)
    elif option == "Preferencias de usuarios":
       return create_user_analysis(df, parent)
    elif option == "Insight":
       return create_insight_analysis(df, parent)
    else:
        # Mensaje de error si la opción no existe
        import tkinter as tk
        tk.Label(parent, text=f"Opción no reconocida: {option}", fg="red", bg="#1a1a2e").pack()
        return False
