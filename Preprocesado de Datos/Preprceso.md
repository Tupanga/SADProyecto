Proyecto: Análisis de Sentimientos - Hinge vs Boo (Tarea A)
Este repositorio contiene la implementación del preprocesado y los modelos de clasificación para el análisis de satisfacción de usuarios de aplicaciones de citas.

Fase 1: Preprocesado de Datos
En esta etapa, se ha realizado una limpieza profunda de los datasets originales de Hinge y Boo para asegurar la calidad de los datos de entrada del modelo predictivo.

Pasos Realizados:
Limpieza de Estructura:

Eliminación de columnas residuales generadas por errores de lectura en los CSV originales (Unnamed: 6, Unnamed: 7).

Filtrado de registros con valores nulos en el campo crítico content.

Estandarización de Etiquetas (Sentimiento):
Siguiendo las instrucciones del jefe de proyecto, se transformó la puntuación numérica (score) en tres categorías discretas:

Negativo: Puntuaciones 1 y 2.

Neutro: Puntuación 3.

Positivo: Puntuaciones 4 y 5.

Normalización de Texto (NLP):
Se aplicó un pipeline de limpieza de texto mediante expresiones regulares (Regex) que incluye:

Conversión de todo el texto a minúsculas.

Eliminación de URLs y enlaces web.

Eliminación de caracteres especiales y puntuación (manteniendo tildes y eñes).

Eliminación de espacios en blanco redundantes.

Análisis de Distribución:
Tras el preprocesado, se detectó un fuerte desbalance de clases que será abordado en la siguiente fase (Balanceo):


Hinge: Predominio de críticas negativas (~7,300 Negativas vs ~800 Neutras).


Boo: Predominio de críticas positivas (~7,400 Positivas vs ~1,000 Negativas).
