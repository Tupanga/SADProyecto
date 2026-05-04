# 📊 Proyecto: Análisis de Sentimientos y Clustering (Boo vs Hinge)

Este repositorio contiene los scripts desarrollados para la Tarea C (Clustering No Supervisado e Insights) del proyecto de Toma de Decisiones. El objetivo es analizar las opiniones preprocesadas de los usuarios de las aplicaciones de citas "Boo" y "Hinge" para extraer los motivos reales de queja y satisfacción.

## 📁 Estructura de Archivos

Asegúrate de que los siguientes archivos de datos (en formato CSV) se encuentran en la misma carpeta que los scripts antes de ejecutar nada:
*   `Boo_preprocesado.csv`
*   `Hinge_preprocesado.csv`

El código está dividido en 4 scripts principales, que representan la evolución metodológica de nuestro análisis:

1.  `clustering.py`: Pipeline básico de K-Means y vectorización TF-IDF.
2.  `hard_c.py`: Implementación definitiva de **Hard Clustering (K-Means)**. Asigna cada comentario a un único clúster y extrae las palabras clave (unigramas) más representativas de cada grupo. Genera los CSVs finales con la columna `cluster_hard`.
3.  `soft_c.py`: Implementación de **Soft Clustering (LDA)** mediante la librería Gensim. Detecta tópicos latentes y asigna probabilidades de pertenencia a cada comentario.
4.  `soft_c_bigramas.py`: Evolución del modelo LDA. Integra detección automática de **Bigramas** (ej. *fake_profiles*, *waste_time*) antes de modelar, logrando un contexto semántico mucho más maduro y exacto para las decisiones de negocio.

## ⚙️ Requisitos Previos (Instalación)

Para ejecutar estos scripts, necesitas tener instalado Python 3 y las siguientes librerías. Puedes instalarlas todas de golpe ejecutando este comando en tu terminal o consola (Command Prompt / PowerShell):
```bash
pip install pandas scikit-learn matplotlib gensim
