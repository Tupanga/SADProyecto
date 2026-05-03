# Memoria Técnica - Tarea A: Clasificación Predictiva de Sentimientos

**Proyecto:** Sistemas de Ayuda a la Decisión - Análisis Competitivo Hinge vs. Boo
**Objetivo:** Desarrollar un modelo de Machine Learning capaz de clasificar opiniones de usuarios en tres sentimientos (Positivo, Negativo, Neutro) maximizando la métrica **Macro F-Score** y superando el modelo base (baseline).

---

## 1. Fase de Preprocesamiento (Data Cleaning)
Antes de entrenar cualquier modelo, el texto crudo debió ser estandarizado. La premisa en NLP (Procesamiento de Lenguaje Natural) es: *"Basura entra, basura sale"*.

**¿Qué hicimos y por qué?**
* **Conversión a minúsculas:** Para que el algoritmo no trate "Horrible" y "horrible" como dos palabras matemáticamente distintas.
* **Eliminación de URLs y menciones:** Mediante Expresiones Regulares (Regex) eliminamos enlaces. *Razón:* Una URL no aporta carga semántica sobre el sentimiento del usuario.
* **Filtro de caracteres especiales y emojis:** Se eliminó todo lo que no fueran letras, números o espacios. *Razón:* Reducir el ruido visual para que la vectorización TF-IDF se centre exclusivamente en el léxico.
* **Stopwords personalizadas:** Se eliminaron palabras vacías del inglés y se añadieron términos del dominio como "app", "dating", "hinge" y "boo". *Razón:* Estas palabras aparecen en el 90% de las opiniones (buenas y malas), por lo que no ayudan al modelo a discriminar el sentimiento y solo consumen memoria.

---

## 2. Registro de Experimentos y Evolución del Modelo

A continuación se detalla el pipeline iterativo de modelado. El progreso se midió utilizando el **Macro F-Score** debido al alto desbalance de clases (muchas opiniones de un tipo, muy pocas del otro).

### Experimento 1: El "Baseline" (Punto de partida)
* **Configuración:** `TfidfVectorizer` (Unigramas, max_features=5000) + Algoritmo `LogisticRegression` clásico. Sin balanceo de datos.
* **Objetivo:** Establecer la marca mínima a superar.
* **Resultado:** **FRACASO (Macro F-Score deficiente).**
* **¿Por qué falló?:** El análisis exploratorio reveló un desbalance extremo (ej. la clase "Negativa" dominaba Hinge, y la clase "Neutra" era casi inexistente). El modelo desarrolló un sesgo hacia la clase mayoritaria (se volvió "perezoso"), prediciendo casi todo como Negativo/Positivo e ignorando los matices de las clases minoritarias, lo que hundió la media del F-Score.

### Experimento 2: Introducción de Balanceo (SMOTE)
* **Configuración:** Igual que el Exp. 1, pero aplicando **SMOTE** (Synthetic Minority Over-sampling Technique).
* **Precaución metodológica:** SMOTE se aplicó *únicamente* sobre el conjunto de entrenamiento (X_train). *Razón:* Si se aplica antes de dividir los datos, ocurre "Data Leakage" (fuga de información) y el test deja de representar la realidad.
* **Resultado:** **ÉXITO (Salto drástico en F-Score).**
* **¿Por qué funcionó?:** Al crear datos sintéticos (matemáticamente similares a los reales) para las clases minoritarias, obligamos al modelo a estudiar los patrones de la clase Neutra con la misma intensidad que las otras. El *Recall* de las clases minoritarias se disparó.

### Experimento 3: Exploración de Algoritmos Tradicionales (KNN)
* **Configuración:** `TfidfVectorizer` + `SMOTE` + `KNeighborsClassifier` (k=5).
* **Objetivo:** Cumplir con la rúbrica de evaluar diferentes algoritmos clásicos.
* **Resultado:** **FRACASO ABSOLUTO (F-Score muy inferior al Baseline).**
* **¿Por qué falló?:** KNN sufre de la *"Maldición de la Dimensionalidad"*. Al transformar el texto en 5.000 características numéricas (features), el espacio geométrico es tan vasto que la distancia euclidiana entre los puntos (opiniones) pierde significado. Para KNN, todos los textos parecían estar a la misma distancia.

### Experimento 4: El Modelo "Sobre-Ingenierizado" (Tomek Links)
* **Configuración:** Introducción de Bigramas `ngram_range=(1,2)` para capturar contexto + Balanceo híbrido `SMOTETomek` (Oversampling + Undersampling) + Algoritmos complejos (Random Forest / Grid Search en RegLog).
* **Objetivo:** Capturar negaciones (ej. "not good") y limpiar las "zonas fronterizas" de datos ruidosos.
* **Resultado:** **FRACASO INESPERADO (El rendimiento bajó respecto al Exp. 2).**
* **¿Por qué falló?:** 1. *Sobrecarga del Vectorizador:* Al añadir bigramas sin aumentar el `max_features` (mantenido en 5000), combinaciones inútiles como "the app" ocuparon el espacio de palabras sueltas vitales.
    2. *Pérdida de contexto humano:* El lenguaje natural es intrínsecamente ambiguo (sarcasmo, opiniones mixtas). La técnica Tomek Links eliminó agresivamente datos en las fronteras de decisión, haciendo que el modelo perdiera la capacidad de entender esa "zona gris" natural de las opiniones reales.

### Experimento 5: 
Tras analizar el fracaso del Experimento 4, diseñamos una arquitectura que combinara lo mejor de cada iteración:
* **Configuración Final:** * `TfidfVectorizer` con Stopwords personalizadas.
    * N-gramas `(1, 2)` (Para entender contextos como "very bad").
    * **`max_features=7000`** (Aumentado para albergar tanto unigramas como los nuevos bigramas sin perder datos clave).
    * Balanceo **SMOTE tradicional** (Sin limpieza destructiva de fronteras).
    * `LogisticRegression` (Algoritmo lineal, excelente manejando alta dimensionalidad sin sobreajuste).

* **Resultado:** **ÉXITO TOTAL.**
* **Conclusión Técnica:** Este modelo alcanzó el Macro F-Score más alto (superando ampliamente el margen requerido sobre el baseline). Demostró que en NLP, un modelo lineal simple bien alimentado (vocabulario amplio + clases balanceadas) generaliza mucho mejor que algoritmos pesados (KNN o Random Forest) que sufren con matrices dispersas.

---

## 3. Despliegue (Pipeline a Producción)
Para garantizar la reproducibilidad y el uso en un entorno de producción, la arquitectura final se encapsuló en un script unificado (`entrenamiento.py`) que exporta el "conocimiento" matemático en dos archivos `.pkl`:
1. El **Vectorizador** (El "diccionario" de 7.000 palabras aprendido).
2. El **Modelo** (Los pesos de decisión de la Regresión Logística).

Ambos archivos son consumidos por el programa `clasificador.py` adjunto en la entrega, permitiendo clasificar nuevas opiniones instantáneamente.
