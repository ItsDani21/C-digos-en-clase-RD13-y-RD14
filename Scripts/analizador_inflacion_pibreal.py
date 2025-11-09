import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.linear_model import LinearRegression

class AnalizadorInflacionPIBReal:

    def __init__(self):
        self.datos_inflacion = None
        self.datos_pib_nominal = None
        self.pib_real_calculado = None

    def cargar_datos_inflacion_ejemplo(self):

        # Datos de inflacion anual (%) - ajustados a la realidad costarricense
        inflacion_data = [
            # Ano, Inflacion_%
            [2018, 2.0], [2019, 2.2], [2020, 0.9], [2021, 3.3],
            [2022, 8.3], [2023, 7.8], [2024, 4.9], [2025, 1.8]
        ]

        self.datos_inflacion = pd.DataFrame(inflacion_data, columns=['Ano', 'Inflacion_Anual_%'])

        return True

    def extraer_pib_nominal_desde_datos(self, datos_pib_gasto):
        """
        Extrae datos del PIB nominal desde el dataset de componentes de gasto
        """
        if datos_pib_gasto is None:
            print("No hay datos de PIB disponibles")
            return None

        try:
            # Procesar la estructura compleja del dataset
            datos = datos_pib_gasto.copy()

            # Encontrar la fila del PIB total
            fila_pib = datos[datos['COMPONENTES'] == 'Producto Interno Bruto']
            if len(fila_pib) == 0:
                print("No se encontro 'Producto Interno Bruto' en los datos")
                return None

            fila_pib = fila_pib.iloc[0]

            # Mapeo de anos y sus columnas
            anos_config = {
                2018: [1, 2, 3, 4],    # Columnas 1-4: 2018
                2019: [5, 6, 7, 8],    # Columnas 5-8: 2019
                2020: [9, 10, 11, 12], # Columnas 9-12: 2020
                2021: [13, 14, 15, 16], # Columnas 13-16: 2021
                2022: [17, 18, 19, 20], # Columnas 17-20: 2022
                2023: [21, 22, 23, 24], # Columnas 21-24: 2023
                2024: [25, 26, 27, 28], # Columnas 25-28: 2024
                2025: [29, 30, 31, 32]  # Columnas 29-32: 2025
            }

            pib_data = []
            trimestres = ['E-M', 'E-J', 'E-S', 'E-D']

            for ano, indices in anos_config.items():
                for i, idx in enumerate(indices):
                    if idx < len(fila_pib):
                        valor = fila_pib.iloc[idx]
                        if pd.notna(valor) and str(valor).strip() != '':
                            try:
                                # Convertir a numerico - estos son indices base 2018
                                valor_indice = float(valor)

                                # Convertir indice a valor nominal aproximado
                                pib_nominal = valor_indice * 10000  # Factor de escala

                                pib_data.append({
                                    'Ano': ano,
                                    'Trimestre': i + 1,
                                    'Trimestre_Texto': trimestres[i],
                                    'PIB_Nominal_Millones': pib_nominal,
                                    'Indice_Base_2018': valor_indice
                                })
                            except (ValueError, TypeError) as e:
                                print(f"Error procesando valor: {valor}, error: {e}")
                                continue

            if not pib_data:
                print("No se pudieron extraer datos del PIB")
                return None

            df_resultado = pd.DataFrame(pib_data)
            print(f"Datos de PIB nominal extraidos: {len(df_resultado)} registros")
            return df_resultado.sort_values(['Ano', 'Trimestre'])

        except Exception as e:
            print(f"Error procesando datos del PIB: {e}")
            return None

    def calcular_pib_real(self, ano_base=2018):
        """
        Calcula el PIB real ajustado por inflacion
        """
        if self.datos_pib_nominal is None or self.datos_inflacion is None:
            print("Faltan datos para calcular PIB real")
            return None

        try:
            # Consolidar datos anuales del PIB nominal (promedio de trimestres)
            pib_anual = self.datos_pib_nominal.groupby('Ano').agg({
                'PIB_Nominal_Millones': 'mean',
                'Indice_Base_2018': 'mean'
            }).reset_index()

            print(f"Anos con datos de PIB: {list(pib_anual['Ano'])}")
            print(f"Anos con datos de inflacion: {list(self.datos_inflacion['Ano'])}")

            # Combinar con datos de inflacion
            datos_combinados = pd.merge(pib_anual, self.datos_inflacion, on='Ano', how='inner')

            if datos_combinados.empty:
                print("No hay anos coincidentes entre PIB e inflacion")
                return None

            print(f"Anos combinados exitosamente: {list(datos_combinados['Ano'])}")

            # Ordenar por ano
            datos_combinados = datos_combinados.sort_values('Ano').reset_index(drop=True)

            # Calcular indice de precios (ano base = 100)
            datos_combinados['Indice_Precios'] = 100.0

            # Encontrar posicion del ano base
            base_mask = datos_combinados['Ano'] == ano_base

            if not base_mask.any():
                print(f"Ano base {ano_base} no encontrado, usando {datos_combinados['Ano'].iloc[0]}")
                ano_base = datos_combinados['Ano'].iloc[0]
                base_mask = datos_combinados['Ano'] == ano_base

            base_idx = datos_combinados[base_mask].index[0]

            # Calcular indice de precios hacia adelante (anos posteriores al base)
            for i in range(base_idx + 1, len(datos_combinados)):
                inflacion = datos_combinados.loc[i, 'Inflacion_Anual_%']
                indice_anterior = datos_combinados.loc[i-1, 'Indice_Precios']
                datos_combinados.loc[i, 'Indice_Precios'] = indice_anterior * (1 + inflacion/100)

            # Calcular indice de precios hacia atras (anos anteriores al base)
            for i in range(base_idx - 1, -1, -1):
                inflacion_siguiente = datos_combinados.loc[i+1, 'Inflacion_Anual_%']
                indice_siguiente = datos_combinados.loc[i+1, 'Indice_Precios']
                datos_combinados.loc[i, 'Indice_Precios'] = indice_siguiente / (1 + inflacion_siguiente/100)

            # Calcular PIB real (PIB nominal ajustado por inflacion)
            datos_combinados['PIB_Real_Millones'] = (
                datos_combinados['PIB_Nominal_Millones'] * 100 / datos_combinados['Indice_Precios']
            )

            # Calcular tasas de crecimiento
            datos_combinados['Crecimiento_Nominal_%'] = datos_combinados['PIB_Nominal_Millones'].pct_change() * 100
            datos_combinados['Crecimiento_Real_%'] = datos_combinados['PIB_Real_Millones'].pct_change() * 100
            datos_combinados['Brecha_Inflacionaria'] = (
                datos_combinados['Crecimiento_Nominal_%'] - datos_combinados['Crecimiento_Real_%']
            )

            self.pib_real_calculado = datos_combinados
            print(f"PIB real calculado exitosamente para {len(datos_combinados)} anos")
            return datos_combinados

        except Exception as e:
            print(f"Error calculando PIB real: {e}")
            return None

    def visualizar_inflacion_pib_real(self):
        """
        Crea visualizaciones para inflacion y PIB real vs nominal
        """
        if self.pib_real_calculado is None:
            print("Primero debe calcular el PIB real")
            return None

        datos = self.pib_real_calculado

        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ANALISIS DE INFLACION Y PIB REAL - COSTA RICA',
                    fontsize=16, fontweight='bold', y=0.95)

        # 1. EVOLUCION DE LA INFLACION
        bars1 = axes[0, 0].bar(datos['Ano'], datos['Inflacion_Anual_%'],
                              color='#FF6B6B', alpha=0.7, edgecolor='darkred')
        axes[0, 0].axhline(y=datos['Inflacion_Anual_%'].mean(), color='red',
                          linestyle='--', alpha=0.7, linewidth=1, label='Promedio')
        axes[0, 0].set_title('Evolucion de la Inflacion Anual', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Ano')
        axes[0, 0].set_ylabel('Inflacion (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Anadir valores en las barras
        for bar, valor in zip(bars1, datos['Inflacion_Anual_%']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{valor:.1f}%', ha='center', va='bottom',
                           fontweight='bold', fontsize=9)

        # 2. COMPARACION PIB NOMINAL VS REAL
        anos = datos['Ano']
        ancho = 0.35
        x_pos = np.arange(len(anos))

        bars_nom = axes[0, 1].bar(x_pos - ancho/2, datos['PIB_Nominal_Millones'] / 1000,
                                 ancho, label='PIB Nominal', alpha=0.7, color='#4ECDC4')
        bars_real = axes[0, 1].bar(x_pos + ancho/2, datos['PIB_Real_Millones'] / 1000,
                                  ancho, label='PIB Real', alpha=0.7, color='#45B7D1')

        axes[0, 1].set_title('Comparacion: PIB Nominal vs PIB Real', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Ano')
        axes[0, 1].set_ylabel('PIB (Miles de millones)')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(anos, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. CRECIMIENTO NOMINAL VS REAL
        mask = datos['Crecimiento_Nominal_%'].notna() & datos['Crecimiento_Real_%'].notna()
        datos_plot = datos[mask]

        if not datos_plot.empty:
            axes[1, 0].plot(datos_plot['Ano'], datos_plot['Crecimiento_Nominal_%'],
                           marker='o', linewidth=2, label='Crecimiento Nominal',
                           color='#FFA07A', markersize=6)
            axes[1, 0].plot(datos_plot['Ano'], datos_plot['Crecimiento_Real_%'],
                           marker='s', linewidth=2, label='Crecimiento Real',
                           color='#20B2AA', markersize=6)

            axes[1, 0].set_title('Crecimiento Economico: Nominal vs Real', fontweight='bold', fontsize=12)
            axes[1, 0].set_xlabel('Ano')
            axes[1, 0].set_ylabel('Crecimiento Anual (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. BRECHA INFLACIONARIA
        if not datos_plot.empty:
            bars_brecha = axes[1, 1].bar(datos_plot['Ano'], datos_plot['Brecha_Inflacionaria'],
                                        color='#FFD700', alpha=0.7, edgecolor='darkorange')
            axes[1, 1].axhline(y=0, color='red', linestyle='-', alpha=0.5)
            axes[1, 1].set_title('Brecha Inflacionaria\n(Diferencia: Nominal - Real)', fontweight='bold', fontsize=12)
            axes[1, 1].set_xlabel('Ano')
            axes[1, 1].set_ylabel('Diferencia en Crecimiento (%)')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            axes[1, 1].tick_params(axis='x', rotation=45)

            # Anadir valores en las barras
            for bar, valor in zip(bars_brecha, datos_plot['Brecha_Inflacionaria']):
                va_pos = 'bottom' if valor >= 0 else 'top'
                offset = 0.1 if valor >= 0 else -0.2
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, valor + offset,
                               f'{valor:.1f}%', ha='center', va=va_pos,
                               fontweight='bold', fontsize=9)

        plt.tight_layout()
        return fig

    def generar_reporte_inflacion(self):
        """
        Genera un reporte detallado del analisis de inflacion
        """
        if self.pib_real_calculado is None:
            print("Primero debe calcular el PIB real")
            return

        datos = self.pib_real_calculado

        print("=" * 80)
        print("INFORME AVANZADO - INFLACION Y PIB REAL")
        print("=" * 80)

        print(f"\nANALISIS DE INFLACION ({datos['Ano'].min()}-{datos['Ano'].max()}):")
        print("-" * 50)
        print(f"Inflacion promedio: {datos['Inflacion_Anual_%'].mean():.2f}%")
        print(f"Inflacion maxima: {datos['Inflacion_Anual_%'].max():.2f}%")
        print(f"Inflacion minima: {datos['Inflacion_Anual_%'].min():.2f}%")

        print(f"\nCOMPARACION PIB NOMINAL VS REAL:")
        print("-" * 60)
        print(f"{'Ano':<6} {'PIB Nominal':<15} {'PIB Real':<15} {'Inflacion':<12} {'Crec. Nominal':<14} {'Crec. Real':<12}")
        print("-" * 80)

        for _, fila in datos.iterrows():
            pib_nom = f"{fila['PIB_Nominal_Millones']/1000:>8.1f}B"
            pib_real = f"{fila['PIB_Real_Millones']/1000:>8.1f}B"
            inflacion = f"{fila['Inflacion_Anual_%']:>6.1f}%"
            crec_nom = f"{fila['Crecimiento_Nominal_%']:>6.1f}%" if not pd.isna(fila['Crecimiento_Nominal_%']) else "   N/A"
            crec_real = f"{fila['Crecimiento_Real_%']:>6.1f}%" if not pd.isna(fila['Crecimiento_Real_%']) else "   N/A"

            print(f"{fila['Ano']:<6} {pib_nom:<15} {pib_real:<15} {inflacion:<12} {crec_nom:<14} {crec_real:<12}")

        # Estadisticas de crecimiento
        datos_crecimiento = datos.dropna(subset=['Crecimiento_Nominal_%', 'Crecimiento_Real_%'])
        if not datos_crecimiento.empty:
            print(f"\nESTADISTICAS DE CRECIMIENTO:")
            print("-" * 40)
            print(f"Crecimiento nominal promedio: {datos_crecimiento['Crecimiento_Nominal_%'].mean():.2f}%")
            print(f"Crecimiento real promedio: {datos_crecimiento['Crecimiento_Real_%'].mean():.2f}%")
            print(f"Brecha inflacionaria promedio: {datos_crecimiento['Brecha_Inflacionaria'].mean():.2f}%")

        print(f"\nINTERPRETACION ECONOMICA:")
        print("-" * 40)
        print("PIB Nominal: Incluye efectos de precios (inflacion)")
        print("PIB Real: Mide crecimiento real de la economia")
        print("Brecha positiva: Inflacion aumenta crecimiento aparente")
        print("Para politica economica: El PIB real es la medida relevante")

        print("\n" + "=" * 80)
        print("ANALISIS DE INFLACION COMPLETADO")
        print("=" * 80)

# CLASE PRINCIPAL ACTUALIZADA
class AnalizadorPIBCompleto:
    """
    Clase principal que integra todas las funcionalidades
    """

    def __init__(self):
        self.datos_pib_gasto = None
        self.datos_imae = None
        self.datos_tasa_crecimiento = None
        self.datos_incidencia = None

    def cargar_datos(self):
        """Carga todos los datasets"""
        try:
            self.datos_pib_gasto = pd.read_csv('../Datasets/pib_gasto_2018.csv', encoding='utf-8')
            self.datos_imae = pd.read_csv('../Datasets/imae_2018.csv', encoding='utf-8')
            self.datos_tasa_crecimiento = pd.read_csv('../Datasets/Tasa de crecimiento.csv', encoding='utf-8')
            self.datos_incidencia = pd.read_csv('../Datasets/INCIDENCIA POR COMPONENTE.csv', encoding='utf-8')

            print("Todos los datasets cargados exitosamente")
            return True
        except Exception as e:
            print(f"Error al cargar: {e}")
            return False

    def ejecutar_analisis_crecimiento(self):
        """Ejecuta solo el analisis de crecimiento (tu codigo existente)"""
        print("\nEJECUTANDO ANALISIS DE CRECIMIENTO...")
        # Aqui iria tu codigo existente de analisis de crecimiento
        print("Analisis de crecimiento completado")

    def ejecutar_analisis_inflacion(self):
        """Ejecuta el analisis avanzado de inflacion"""
        print("\nEJECUTANDO ANALISIS DE INFLACION (Nivel Avanzado)...")

        # Inicializar analizador de inflacion
        analizador_inflacion = AnalizadorInflacionPIBReal()

        # Cargar datos de inflacion (ejemplo)
        analizador_inflacion.cargar_datos_inflacion_ejemplo()

        # Extraer datos de PIB nominal
        datos_pib_nominal = analizador_inflacion.extraer_pib_nominal_desde_datos(self.datos_pib_gasto)

        if datos_pib_nominal is not None:
            analizador_inflacion.datos_pib_nominal = datos_pib_nominal

            # Calcular PIB real
            pib_real = analizador_inflacion.calcular_pib_real(ano_base=2018)

            if pib_real is not None:
                # Generar reporte
                analizador_inflacion.generar_reporte_inflacion()

                # Crear visualizaciones
                print("\nGenerando visualizaciones de inflacion...")
                fig = analizador_inflacion.visualizar_inflacion_pib_real()

                # Guardar y mostrar
                plt.savefig('analisis_inflacion_pib_real.png', dpi=300, bbox_inches='tight')
                print("Visualizacion guardada como 'analisis_inflacion_pib_real.png'")
                plt.show()

                # Exportar datos
                pib_real.to_csv('resultados_pib_real.csv', index=False)
                print("Datos exportados a 'resultados_pib_real.csv'")
            else:
                print("No se pudo calcular el PIB real")
        else:
            print("No se pudieron extraer datos del PIB nominal")

# EJECUCION SIMPLIFICADA
if __name__ == "__main__":
    analizador = AnalizadorPIBCompleto()

    if analizador.cargar_datos():
        # Ejecutar analisis de inflacion (funcionalidad nueva)
        analizador.ejecutar_analisis_inflacion()
    else:
        print("No se pudieron cargar los datos")