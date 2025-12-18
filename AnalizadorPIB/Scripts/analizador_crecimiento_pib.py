import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression

class AnalizadorCrecimientoPIB_Final:
    """
    Versión final específicamente adaptada a la estructura real de los datasets
    """

    def __init__(self):
        self.datos_pib_gasto = None
        self.datos_imae = None
        self.datos_tasa_crecimiento = None
        self.datos_incidencia = None

    def cargar_datos(self):
        """Carga todos los datasets"""
        try:
            self.datos_pib_gasto = pd.read_csv('Datasets/pib_gasto_2018.csv', encoding='utf-8')
            self.datos_imae = pd.read_csv('Datasets/imae_2018.csv', encoding='utf-8')
            self.datos_tasa_crecimiento = pd.read_csv('Datasets/Tasa de crecimiento.csv', encoding='utf-8')
            self.datos_incidencia = pd.read_csv('Datasets/INCIDENCIA POR COMPONENTE.csv', encoding='utf-8')

            print("Todos los datasets cargados exitosamente")
            return True
        except Exception as e:
            print(f"Error al cargar: {e}")
            return False

    def procesar_pib_gasto(self):
        """
        Procesa el dataset complejo de PIB por componentes de gasto
        Estructura: Cada año tiene 4 columnas (trimestres: E-M, E-J, E-S, E-D)
        """
        if self.datos_pib_gasto is None:
            return None

        datos = self.datos_pib_gasto.copy()

        # Eliminar filas vacías
        datos = datos.dropna(subset=['COMPONENTES'])
        datos = datos[datos['COMPONENTES'].str.strip() != '']

        # Mapeo de años y sus columnas correspondientes
        años_config = {
            '2018': [1, 2, 3, 4],    # Columnas 1-4: 2018 E-M, E-J, E-S, E-D
            '2019': [5, 6, 7, 8],    # Columnas 5-8: 2019
            '2020': [9, 10, 11, 12], # Columnas 9-12: 2020
            '2021': [13, 14, 15, 16], # Columnas 13-16: 2021
            '2022': [17, 18, 19, 20], # Columnas 17-20: 2022
            '2023': [21, 22, 23, 24], # Columnas 21-24: 2023
            '2024': [25, 26, 27, 28], # Columnas 25-28: 2024
            '2025': [29, 30, 31, 32]  # Columnas 29-32: 2025
        }

        trimestres = ['E-M', 'E-J', 'E-S', 'E-D']
        datos_largos = []

        for _, fila in datos.iterrows():
            componente = fila['COMPONENTES']

            for año, indices in años_config.items():
                for i, idx in enumerate(indices):
                    if idx < len(fila):
                        valor = fila.iloc[idx]
                        if pd.notna(valor) and str(valor).strip() != '':
                            try:
                                datos_largos.append({
                                    'Año': int(año),
                                    'Trimestre': i + 1,
                                    'Trimestre_Texto': trimestres[i],
                                    'Componente': componente,
                                    'Valor_Indice': float(valor)
                                })
                            except (ValueError, TypeError):
                                continue

        df_resultado = pd.DataFrame(datos_largos)

        # Filtrar solo componentes relevantes
        componentes_relevantes = [
            'Consumo Final', 'Consumo Privado', 'Consumo Público',
            'Formación Bruta de Capital Fijo', 'Exportaciones', 'Importaciones',
            'Producto Interno Bruto'
        ]
        df_resultado = df_resultado[df_resultado['Componente'].isin(componentes_relevantes)]

        return df_resultado.sort_values(['Año', 'Trimestre'])

    def calcular_crecimiento_anual_pib(self):
        """Calcula crecimiento anual del PIB desde datos de componentes"""
        datos_pib = self.procesar_pib_gasto()
        if datos_pib is None:
            return None

        # Filtrar solo el PIB total
        pib_total = datos_pib[datos_pib['Componente'] == 'Producto Interno Bruto'].copy()

        # Calcular crecimiento anual (mismo trimestre vs año anterior)
        pib_total = pib_total.sort_values(['Trimestre', 'Año'])
        pib_total['Valor_Anterior'] = pib_total.groupby('Trimestre')['Valor_Indice'].shift(1)
        pib_total['Crecimiento_Anual_%'] = (
            (pib_total['Valor_Indice'] - pib_total['Valor_Anterior']) /
            pib_total['Valor_Anterior'] * 100
        )

        return pib_total.dropna(subset=['Crecimiento_Anual_%'])

    def calcular_crecimiento_trimestral_pib(self):
        """Calcula crecimiento trimestral del PIB"""
        datos_pib = self.procesar_pib_gasto()
        if datos_pib is None:
            return None

        pib_total = datos_pib[datos_pib['Componente'] == 'Producto Interno Bruto'].copy()
        pib_total = pib_total.sort_values(['Año', 'Trimestre'])

        # Crecimiento trimestre a trimestre
        pib_total['Valor_Anterior_Trim'] = pib_total['Valor_Indice'].shift(1)
        pib_total['Crecimiento_Trimestral_%'] = (
            (pib_total['Valor_Indice'] - pib_total['Valor_Anterior_Trim']) /
            pib_total['Valor_Anterior_Trim'] * 100
        )

        return pib_total.dropna(subset=['Crecimiento_Trimestral_%'])

    def analizar_imae_crecimiento(self):
        """Analiza crecimiento usando IMAE (datos mensuales confiables)"""
        if self.datos_imae is None:
            return None

        datos = self.datos_imae.copy()

        # Usar datos desestacionalizados (más confiables)
        datos['Crecimiento_Interanual_%'] = pd.to_numeric(
            datos['Interanual Desestacionalizada'], errors='coerce'
        )

        # Calcular promedios anuales
        crecimiento_anual = datos.groupby('Año').agg({
            'Crecimiento_Interanual_%': 'mean'
        }).reset_index()

        # Calcular crecimiento trimestral desde mensual
        mapeo_mes_trimestre = {
            'Enero': 1, 'Febrero': 1, 'Marzo': 1,
            'Abril': 2, 'Mayo': 2, 'Junio': 2,
            'Julio': 3, 'Agosto': 3, 'Septiembre': 3,
            'Octubre': 4, 'Noviembre': 4, 'Diciembre': 4
        }

        datos['Trimestre'] = datos['Mes'].map(mapeo_mes_trimestre)
        crecimiento_trimestral = datos.groupby(['Año', 'Trimestre']).agg({
            'Crecimiento_Interanual_%': 'mean'
        }).reset_index()

        return {
            'anual': crecimiento_anual,
            'trimestral': crecimiento_trimestral,
            'mensual': datos
        }

    def analizar_tasas_historicas(self):
        """Analiza tasas de crecimiento históricas"""
        if self.datos_tasa_crecimiento is None:
            return None

        datos = self.datos_tasa_crecimiento.copy()

        # Usar PIB Referencia 2018 (serie más consistente)
        datos_limpios = datos[['Período', 'PIB Referencia 2018']].copy()
        datos_limpios.columns = ['Año', 'Crecimiento_Anual_%']
        datos_limpios['Crecimiento_Anual_%'] = pd.to_numeric(
            datos_limpios['Crecimiento_Anual_%'], errors='coerce'
        )

        # Estadísticas
        stats = {
            'crecimiento_promedio': datos_limpios['Crecimiento_Anual_%'].mean(),
            'crecimiento_max': datos_limpios['Crecimiento_Anual_%'].max(),
            'crecimiento_min': datos_limpios['Crecimiento_Anual_%'].min(),
            'volatilidad': datos_limpios['Crecimiento_Anual_%'].std(),
            'ultimos_5_años': datos_limpios.tail(5)['Crecimiento_Anual_%'].mean()
        }

        return {
            'datos': datos_limpios.dropna(),
            'estadisticas': stats
        }

    def generar_tabla_resumen_completa(self):
        """Genera tabla resumen unificando todas las fuentes"""
        resumen_data = []

        # 1. Desde PIB componentes de gasto (trimestral)
        crecimiento_pib = self.calcular_crecimiento_anual_pib()
        if crecimiento_pib is not None:
            for año in crecimiento_pib['Año'].unique():
                datos_año = crecimiento_pib[crecimiento_pib['Año'] == año]
                crecimiento_promedio = datos_año['Crecimiento_Anual_%'].mean()

                resumen_data.append({
                    'Año': año,
                    'Fuente': 'PIB_Componentes',
                    'Crecimiento_%': round(crecimiento_promedio, 2)
                })

        # 2. Desde IMAE
        analisis_imae = self.analizar_imae_crecimiento()
        if analisis_imae is not None:
            for _, fila in analisis_imae['anual'].iterrows():
                if pd.notna(fila['Crecimiento_Interanual_%']):
                    resumen_data.append({
                        'Año': fila['Año'],
                        'Fuente': 'IMAE',
                        'Crecimiento_%': round(fila['Crecimiento_Interanual_%'], 2)
                    })

        # 3. Desde datos históricos
        tasas_historicas = self.analizar_tasas_historicas()
        if tasas_historicas is not None:
            for _, fila in tasas_historicas['datos'].iterrows():
                resumen_data.append({
                    'Año': fila['Año'],
                    'Fuente': 'Historico',
                    'Crecimiento_%': round(fila['Crecimiento_Anual_%'], 2)
                })

        df_resumen = pd.DataFrame(resumen_data)

        # Agregar estadísticas por año
        stats_por_año = df_resumen.groupby('Año').agg({
            'Crecimiento_%': ['mean', 'std', 'count']
        }).round(2)

        return {
            'resumen_detallado': df_resumen,
            'stats_por_año': stats_por_año
        }

    def visualizar_analisis_completo(self):
        """Crea visualizaciones completas del análisis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ANÁLISIS COMPLETO DE CRECIMIENTO ECONÓMICO',
                    fontsize=16, fontweight='bold', y=0.98)

        # 1. COMPARACIÓN DE FUENTES (Años recientes)
        resumen_completo = self.generar_tabla_resumen_completa()
        if not resumen_completo['resumen_detallado'].empty:
            datos_resumen = resumen_completo['resumen_detallado']
            años_recientes = [year for year in range(2018, 2024)]
            datos_recientes = datos_resumen[datos_resumen['Año'].isin(años_recientes)]

            if not datos_recientes.empty:
                pivot_data = datos_recientes.pivot(index='Año', columns='Fuente', values='Crecimiento_%')
                pivot_data.plot(kind='bar', ax=axes[0,0],
                              color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
                axes[0,0].set_title('Comparación de Crecimiento por Fuente\n(Años 2018-2023)',
                                  fontweight='bold', fontsize=12)
                axes[0,0].set_ylabel('Crecimiento Anual (%)')
                axes[0,0].tick_params(axis='x', rotation=45)
                axes[0,0].grid(True, alpha=0.3, axis='y')
                axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
                axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 2. CRECIMIENTO TRIMESTRAL PIB
        crecimiento_trimestral = self.calcular_crecimiento_trimestral_pib()
        if crecimiento_trimestral is not None:
            ultimos_12_trimestres = crecimiento_trimestral.tail(12)

            x_positions = range(len(ultimos_12_trimestres))
            colores = ['green' if x >= 0 else 'red' for x in ultimos_12_trimestres['Crecimiento_Trimestral_%']]

            bars = axes[0,1].bar(x_positions, ultimos_12_trimestres['Crecimiento_Trimestral_%'],
                               color=colores, alpha=0.7)

            axes[0,1].set_title('Crecimiento Trimestral del PIB\n(Últimos 12 trimestres)',
                              fontweight='bold', fontsize=12)
            axes[0,1].set_ylabel('Crecimiento Trimestral (%)')
            axes[0,1].set_xticks(x_positions)
            axes[0,1].set_xticklabels([f"{fila['Año']}-T{fila['Trimestre']}"
                                     for _, fila in ultimos_12_trimestres.iterrows()],
                                    rotation=45, fontsize=9)
            axes[0,1].grid(True, alpha=0.3, axis='y')
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)

            # Añadir valores
            for i, (bar, valor) in enumerate(zip(bars, ultimos_12_trimestres['Crecimiento_Trimestral_%'])):
                axes[0,1].text(bar.get_x() + bar.get_width()/2,
                              bar.get_height() + (0.1 if valor >= 0 else -0.8),
                              f'{valor:.1f}%', ha='center', va='bottom' if valor >= 0 else 'top',
                              fontweight='bold', fontsize=8)

        # 3. EVOLUCIÓN COMPONENTES PIB
        datos_pib = self.procesar_pib_gasto()
        if datos_pib is not None:
            componentes_visualizar = ['Consumo Privado', 'Formación Bruta de Capital Fijo',
                                    'Exportaciones', 'Producto Interno Bruto']
            datos_filtrados = datos_pib[datos_pib['Componente'].isin(componentes_visualizar)]

            # Promedio anual por componente
            datos_anuales = datos_filtrados.groupby(['Año', 'Componente'])['Valor_Indice'].mean().reset_index()

            for componente in componentes_visualizar:
                datos_comp = datos_anuales[datos_anuales['Componente'] == componente]
                axes[1,0].plot(datos_comp['Año'], datos_comp['Valor_Indice'],
                              marker='o', linewidth=2, label=componente, markersize=4)

            axes[1,0].set_title('Evolución de Componentes del PIB\n(Índice Base 2018)',
                              fontweight='bold', fontsize=12)
            axes[1,0].set_xlabel('Año')
            axes[1,0].set_ylabel('Índice (2018 = 100)')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            axes[1,0].grid(True, alpha=0.3)

        # 4. ESTADÍSTICAS HISTÓRICAS
        tasas_historicas = self.analizar_tasas_historicas()
        if tasas_historicas is not None:
            stats = tasas_historicas['estadisticas']

            metricas = ['Promedio\nGeneral', 'Máximo\nHistórico', 'Mínimo\nHistórico', 'Volatilidad', 'Promedio\nÚltimos 5 años']
            valores = [
                stats['crecimiento_promedio'],
                stats['crecimiento_max'],
                stats['crecimiento_min'],
                stats['volatilidad'],
                stats['ultimos_5_años']
            ]
            colores_barras = ['blue', 'green', 'red', 'orange', 'purple']

            bars = axes[1,1].bar(metricas, valores, color=colores_barras, alpha=0.7)
            axes[1,1].set_title('Estadísticas de Crecimiento Histórico\n(1992-2024)',
                              fontweight='bold', fontsize=12)
            axes[1,1].set_ylabel('Crecimiento (%)')
            axes[1,1].grid(True, alpha=0.3, axis='y')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)

            # Añadir valores en las barras
            for bar, valor in zip(bars, valores):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                              f'{valor:.2f}%', ha='center', va='bottom',
                              fontweight='bold', fontsize=9)

        plt.tight_layout()
        return fig

    def generar_reporte_analisis(self):
        """Genera reporte completo del análisis"""
        print("=" * 80)
        print(" INFORME COMPLETO - ANÁLISIS DE CRECIMIENTO DEL PIB")
        print("=" * 80)

        # 1. CRECIMIENTO DESDE PIB COMPONENTES
        crecimiento_pib = self.calcular_crecimiento_anual_pib()
        if crecimiento_pib is not None:
            print("\n CRECIMIENTO ANUAL DEL PIB (Componentes de Gasto):")
            print("-" * 55)
            print(f"{'Año':<8} {'Crecimiento Promedio (%)':<25} {'Trimestres':<15}")
            print("-" * 55)

            for año in sorted(crecimiento_pib['Año'].unique()):
                datos_año = crecimiento_pib[crecimiento_pib['Año'] == año]
                crecimiento_prom = datos_año['Crecimiento_Anual_%'].mean()
                n_trimestres = len(datos_año)
                print(f"{año:<8} {crecimiento_prom:>18.2f}% {n_trimestres:>14}")

        # 2. CRECIMIENTO TRIMESTRAL
        crecimiento_trimestral = self.calcular_crecimiento_trimestral_pib()
        if crecimiento_trimestral is not None:
            print(f"\n CRECIMIENTO TRIMESTRAL RECIENTE:")
            print("-" * 45)
            ultimos_4 = crecimiento_trimestral.tail(4)
            for _, fila in ultimos_4.iterrows():
                tendencia = "↗" if fila['Crecimiento_Trimestral_%'] > 0 else "↘"
                print(f"  {fila['Año']}-T{fila['Trimestre']}: {fila['Crecimiento_Trimestral_%']:6.2f}% {tendencia}")

        # 3. ESTADÍSTICAS HISTÓRICAS
        tasas_historicas = self.analizar_tasas_historicas()
        if tasas_historicas is not None:
            stats = tasas_historicas['estadisticas']
            print(f"\n ESTADÍSTICAS HISTÓRICAS (1992-2024):")
            print("-" * 45)
            print(f"  • Crecimiento promedio: {stats['crecimiento_promedio']:7.2f}%")
            print(f"  • Crecimiento máximo:   {stats['crecimiento_max']:7.2f}%")
            print(f"  • Crecimiento mínimo:   {stats['crecimiento_min']:7.2f}%")
            print(f"  • Volatilidad:          {stats['volatilidad']:7.2f}%")
            print(f"  • Promedio últimos 5 años: {stats['ultimos_5_años']:7.2f}%")

        # 4. COMPARACIÓN ENTRE FUENTES
        resumen_completo = self.generar_tabla_resumen_completa()
        if not resumen_completo['resumen_detallado'].empty:
            print(f"\n COMPARACIÓN ENTRE FUENTES DE DATOS:")
            print("-" * 55)
            datos_resumen = resumen_completo['resumen_detallado']
            años_comparar = [2021, 2022, 2023]

            for año in años_comparar:
                datos_año = datos_resumen[datos_resumen['Año'] == año]
                if not datos_año.empty:
                    print(f"\n  Año {año}:")
                    for _, fila in datos_año.iterrows():
                        print(f"    • {fila['Fuente']}: {fila['Crecimiento_%']:6.2f}%")


# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Inicializar y ejecutar análisis
    analizador = AnalizadorCrecimientoPIB_Final()

    if analizador.cargar_datos():
        print("\n" + " Procesando datos..." + "\n")

        # Generar reporte completo
        analizador.generar_reporte_analisis()

        # Crear visualizaciones
        print("\n Generando visualizaciones...")
        figura = analizador.visualizar_analisis_completo()

        # Guardar figura
        plt.savefig('analisis_crecimiento_pib.png', dpi=300, bbox_inches='tight')
        print("Visualización guardada como 'analisis_crecimiento_pib.png'")

        # Mostrar figura
        plt.show()

        # Exportar datos procesados
        try:
            crecimiento_anual = analizador.calcular_crecimiento_anual_pib()
            if crecimiento_anual is not None:
                crecimiento_anual.to_csv('resultados_crecimiento_anual.csv', index=False)
                print(" Datos exportados a 'resultados_crecimiento_anual.csv'")
        except Exception as e:
            print(f" Nota: {e}")