# SAMSUNG SIC: Analizador del PIB y Crecimiento Económico

## Descripción del Proyecto

Este proyecto es una herramienta integral desarrollada en Python para el análisis y visualización de datos del Producto Interno Bruto (PIB) y crecimiento económico. Diseñado específicamente para facilitar el aprendizaje de conceptos macroeconómicos, permite a estudiantes y profesionales explorar tendencias económicas sin requerir conocimientos avanzados en economía.

## Objetivos

- **Educativo**: Facilitar la comprensión de conceptos macroeconómicos
- **Analítico**: Proporcionar herramientas para el análisis de datos económicos
- **Visual**: Generar visualizaciones claras de tendencias económicas
- **Práctico**: Ofrecer una interfaz amigable para el análisis interactivo

## Funcionalidades Principales

### 1. **Análisis de Crecimiento del PIB**

- Cálculo de tasas de crecimiento anual del PIB
- Análisis de tendencias históricas
- Comparación de períodos económicos

### 2. **Procesamiento de Datos Económicos**

- Carga y limpieza de datasets económicos
- Integración de múltiples fuentes de datos
- Validación y preparación de datos para análisis

### 3. **Visualización de Tendencias Económicas**

- Gráficos de evolución del PIB temporal
- Análisis comparativo entre componentes económicos
- Visualización de tasas de crecimiento

### 4. **Análisis de Inflación y PIB Real** (Opcional)

- Cálculo del PIB ajustado por inflación
- Análisis del poder adquisitivo real
- Comparativa entre PIB nominal y real

## Estructura del Proyecto

```
Analizador_PIB/
├── 📁 Datasets/                 # Conjuntos de datos económicos
│   ├── pib_gasto_2018.csv
│   ├── lme_2018.csv
│   ├── tasa_crecimiento.csv
│   └── incidencia_componente.csv
├── 📁 Scripts/                  # Módulos de análisis
│   ├── analizador_crecimiento_pib.py
│   ├── analizador_inflacion_pibreal.py
│   └── analizador_pib_completo.py
├── 📁 .venv/                    # Entorno virtual Python
├── 🐍 main.py                   # Aplicación principal con interfaz gráfica
├── 📄 requirements.txt          # Dependencias del proyecto
└── 📄 README.md                 # Documentación
```

## Características Técnicas

### **Tecnologías Utilizadas**

- **Python 3.x** - Lenguaje principal
- **Pandas** - Procesamiento y análisis de datos
- **Matplotlib & Seaborn** - Visualización de datos
- **Scikit-learn** - Modelos de regresión y análisis predictivo
- **Tkinter** - Interfaz gráfica de usuario
- **NumPy** - Cálculos numéricos avanzados

### **Módulos Principales**

#### `AnalizadorCrecimientoPIB_Final`

- Carga y procesamiento de datasets económicos
- Cálculo de tasas de crecimiento
- Análisis de componentes del PIB

#### `AnalizadorInflacionPIBReal`

- Ajuste del PIB por inflación
- Cálculo del PIB real vs nominal
- Análisis del poder adquisitivo

#### `InterfazAnalisisPIB`

- Interfaz gráfica unificada
- Visualización interactiva de datos
- Panel de control económico completo

## Habilidades Desarrolladas

### **Habilidades de Programación**

- **Análisis de Datos**: Uso de Pandas para limpieza y análisis
- **Series Temporales**: Trabajo con datos indexados por fecha
- **Visualización**: Creación de gráficos informativos y claros
- **Interfaces de Usuario**: Desarrollo de GUI con Tkinter

### **Conceptos Económicos**

- Crecimiento económico y tasas de variación
- Componentes del PIB y su incidencia
- Inflación y ajustes del PIB real
- Tendencias macroeconómicas

## Aplicaciones Educativas

Este proyecto es ideal para:

- **Estudiantes de Economía**: Práctica con datos reales
- **Cursos de Macroeconomía**: Herramienta de enseñanza visual
- **Investigación Económica**: Análisis rápido de tendencias
- **Programadores**: Aprendizaje de análisis de datos aplicado

## Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`

### Ejecución

```bash
python main.py
```

## Salidas y Resultados

El proyecto genera:

- **Gráficos interactivos** de tendencias económicas
- **Reportes analíticos** de crecimiento e inflación
- **Dashboard económico** con métricas clave
- **Exportación de resultados** en múltiples formatos

---

_Proyecto diseñado para fines educativos y de análisis económico, facilitando la comprensión de conceptos macroeconómicos complejos mediante herramientas visuales e interactivas._
