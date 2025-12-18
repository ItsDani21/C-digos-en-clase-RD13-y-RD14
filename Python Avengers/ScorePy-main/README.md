<div align="center">

  # ScorePy  
  #### Herramienta de Análisis Educativo
  
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/scorepydark.png" />
    <source media="(prefers-color-scheme: light)" srcset="assets/scorepylight.jpg" />
    <img alt="ScorePy Banner" src="assets/scorepydark.png" width="100%">
  </picture>
  
  [![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads)
  [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
  [![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

## Introducción

**ScorePy** es una herramienta desarrollada en **Python** enfocada en el **análisis de datos del rendimiento estudiantil** a partir de los hábitos y métodos de estudio.

Permite a instituciones académicas obtener **informes detallados y visuales** sobre el desempeño de los estudiantes, generando análisis automáticos y reportes interpretativos listos para presentación.

## Tipos de Análisis Incluidos

Cada análisis se basa en datos recopilados desde formularios personalizados, con la opción de seleccionar métricas específicas según las necesidades del usuario.

- Relación entre horas de estudio y calificación  
- Relación entre método de estudio y calificación  
- Promedio por método de estudio  
- Promedio por rango de horas de estudio  
- Impacto de distracciones en el rendimiento  
- Relación motivación vs promedio general  
- Recursos de estudio más utilizados  
- Frecuencia de repasos por semana  
- Sentimiento del estudiante según su promedio  

> Debajo de cada reporte se genera un comentario automático que destaca el factor con mayor impacto.

## Estructura del Repositorio

```bash
ScorePy/
│
├── 📂 src/
│   ├── analisis_datos.py
│   ├── generar_html.py
│   ├── graficos.py
│   ├── interfaz.py
│   ├── procesamiento.py
│   └── img/
│       ├── archivos.png
│       ├── destino.png
│       ├── footer.png
│       ├── header.png
│       ├── reportes.png
│       └── svg.png
│
├── main.py
├── requirements.txt
├── python.ico
└── README.md
```

## Instalación y Ejecución

```bash
# Clonar el repositorio
git clone https://github.com/miwelo/Analisis-Educativo.git

# Entrar al directorio
cd Analisis-Educativo

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar el programa
python main.py
```

## Demostración Visual

<p align="center">
  <img src="assets/panel.png" alt="Panel principal" width="90%">
</p>

<p align="center">
  <img src="assets/reporte1.png" alt="Ejemplo Reporte 1" width="45%">
  <img src="assets/reporte2.png" alt="Ejemplo Reporte 2" width="45%">
</p>

<p align="center">
  <img src="assets/reporte3.png" alt="Ejemplo Reporte 3" width="45%">
  <img src="assets/reporte4.png" alt="Ejemplo Reporte 4" width="45%">
</p>

<p align="center">
  <img src="assets/reporte5.png" alt="Ejemplo Reporte 5" width="90%">
</p>

## 👥 Autores

<div align="center">
  <a href="https://github.com/miwelo"> <img src="https://github.com/miwelo.png" width="100" height="100" style="border-radius: 50%;" alt="Miguel Inoa"> </a>
  <a href="https://github.com/aslyninmateo"> <img src="https://github.com/aslyninmateo.png" width="100" height="100" style="border-radius: 50%;" alt="Asly Nin"> </a>
  <a href="https://github.com/francisco983412"> <img src="https://github.com/francisco983412.png" width="100" height="100" style="border-radius: 50%;" alt="Francisco Concepción"> </a>
  <a href="https://github.com/BrahyanPro"> <img src="https://github.com/BrahyanPro.png" width="100" height="100" style="border-radius: 50%;" alt="Brahyan Martínez"> </a>
  <a href="https://github.com/Melquiiii11"> <img src="https://github.com/Melquiiii11.png" width="100" height="100" style="border-radius: 50%;" alt="Angel Díaz"> </a>
</div>



