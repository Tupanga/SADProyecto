# Proyecto: Análisis de Sentimientos - Hinge vs Boo (Tarea A)

Este repositorio contiene la implementación del clasificador predictivo para la Tarea A del proyecto de Sistemas de Ayuda a la Decisión.

## 📁 Estructura de Archivos
- `clasificador.py`: Script principal para clasificar nuevas reviews.
- `entrenamiento.py`: Script para generar los modelos a partir de los csv originales.
- `config.json`: Archivo de configuración con las rutas de modelos y datos.
- `requirements.txt`: Librerías necesarias.
- `modelo_entrenado.pkl`: Pesos del modelo Logistic Regression (Generar en Colab).
- `vectorizador.pkl`: Transformador TF-IDF (Generar en Colab).

## 🚀 Ejecución
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar: `python entrenamiento.py`
3. Configurar el archivo `config.json` con la ruta del CSV a analizar y cambiar los nombres por los nombres de los pkls que quieras analizar  .
4. Ejecutar: `python clasificador.py`

## 📊 Metodología (Resumen)
- **Preprocesado**: Limpieza Regex, minúsculas, eliminación de URLs y caracteres especiales.
- **Vectorización**: TF-IDF con 7000 características (unigramas).
- **Balanceo**: SMOTE (Synthetic Minority Over-sampling Technique).
- **Algoritmo**: Regresión Logística (Optimizado).
