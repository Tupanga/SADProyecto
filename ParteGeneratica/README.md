# Parte Generativa — Sentiment Analysis con Ollama

## Descripción

Script de clasificación de sentimientos mediante **prompt engineering** sobre modelos de lenguaje locales a través de **Ollama**. Dado el texto de una reseña de una app de citas, predice si el sentimiento es **positive**, **negative** o **neutral**.

Forma parte del proyecto *Toma de decisiones: ¿Podemos mejorar las valoraciones de nuestra aplicación?* de la asignatura Sistemas de Ayuda a la Decisión.

---

## Requisitos previos

### 1. Instalar Ollama
Descárgalo desde https://ollama.com/ e instálalo en tu sistema.

### 2. Descargar un modelo
```bash
ollama pull mistral       # Recomendado (mejor calidad, ~4GB)
ollama pull llama3        # Alternativa (~4GB)
ollama pull gemma:2b      # Más ligero pero menor precisión (~1.5GB)
```

Verifica que Ollama está corriendo:
```bash
ollama list
```

### 3. Instalar dependencias Python
```bash
pip install -r requirements.txt
```

---

## Uso

### Comando básico
```bash
python sentiment_ollama.py --input Hinge_preprocesado.csv
```

### Todos los argumentos disponibles

| Argumento             | Por defecto              | Descripción                                             |
|-----------------------|--------------------------|---------------------------------------------------------|
| `--input`             | `Hinge_preprocesado.csv` | CSV preprocesado de entrada                             |
| `--output_dir`        | `output/`                | Carpeta donde se guardan los resultados                 |
| `--model`             | `gemma:2b`               | Modelo de Ollama a usar                                 |
| `--strategy`          | `None` (todas)           | Estrategia de prompting a usar                          |
| `--limit`             | `None` (todas)           | Nº máximo de ejemplos del dev a clasificar              |
| `--n_paraphrases`     | `2`                      | Nº de paráfrasis por muestra en oversampling generativo |
| `--skip_oversampling` | `False`                  | Saltar el oversampling generativo                       |
| `--debug`             | `False`                  | Mostrar respuesta raw del modelo para cada reseña       |

### Estrategias de prompting disponibles

| Estrategia          | Descripción                                          |
|---------------------|------------------------------------------------------|
| `zero_shot`         | Solo instrucción directa, sin ejemplos               |
| `one_shot`          | Un ejemplo de referencia                             |
| `few_shot`          | Cinco ejemplos variados del dominio                  |
| `chain_of_thought`  | Razonamiento paso a paso antes de dar la predicción  |

---

## Ejemplos de ejecución

### Prueba rápida (10 ejemplos, ver respuestas del modelo)
```bash
python sentiment_ollama.py --input Hinge_preprocesado.csv --model mistral --strategy few_shot --limit 10 --skip_oversampling --debug
```

### Ejecución para entregar (150 ejemplos, few-shot, sin oversampling)
```bash
python sentiment_ollama.py --input Hinge_preprocesado.csv --model mistral --strategy few_shot --limit 150 --skip_oversampling
```

### Evaluar las 4 estrategias y elegir la mejor automáticamente
```bash
python sentiment_ollama.py --input Hinge_preprocesado.csv --model mistral --limit 150 --skip_oversampling
```

### Ejecución completa con oversampling generativo
```bash
python sentiment_ollama.py --input Hinge_preprocesado.csv --model mistral --strategy few_shot
```

---

## Estructura del CSV de entrada

El script detecta automáticamente las columnas. Estructura esperada:

```
reviewId,content,score,gender,location,date,sentimiento,content_limpio
34c160a2-...,Very bad app fake hai ye,1.0,male,"Rabat, Morocco",2020-10-16,Negativo,very bad app fake hai ye
```

| Columna          | Uso                                                  |
|------------------|------------------------------------------------------|
| `content_limpio` | Texto preprocesado que se envía al modelo            |
| `sentimiento`    | Etiqueta gold (`Positivo`, `Negativo`, `Neutro`)     |
| `score`          | Fallback numérico si no existe `sentimiento`         |

La columna `sentimiento` se mapea internamente:
- `Positivo` → `positive`
- `Negativo` → `negative`
- `Neutro`   → `neutral`

El script divide automáticamente el CSV en **80% train / 20% dev** de forma estratificada.

---

## Ficheros de salida

Todos los ficheros se generan en la carpeta `output/`:

| Fichero                | Contenido                                                           |
|------------------------|---------------------------------------------------------------------|
| `predictions_best.csv` | Texto + etiqueta original + predicción + acierto (SI/NO)           |
| `prompts_log.csv`      | Estrategias probadas con métricas, prompt y ejemplos E/S           |
| `confusion_matrix.csv` | Matriz de confusión de la mejor estrategia                         |
| `paraphrases.csv`      | Dataset aumentado con oversampling generativo                      |
| `train_split.csv`      | 80% del CSV original                                               |
| `dev_split.csv`        | 20% del CSV original                                               |
| `summary.json`         | Resumen: modelo, mejor estrategia, Macro-F1, muestras generadas    |

### Ejemplo de `predictions_best.csv`

| text | etiqueta_original | prediccion | acierto |
|------|-------------------|------------|---------|
| very bad app fake hai ye | negative | negative | SI |
| love this app so much | positive | neutral | NO |

### Ejemplo de `confusion_matrix.csv`

| real_vs_predicho | POSITIVE | NEGATIVE | NEUTRAL |
|------------------|----------|----------|---------|
| POSITIVE         | 45       | 3        | 2       |
| NEGATIVE         | 1        | 38       | 4       |
| NEUTRAL          | 5        | 2        | 12      |

---

## Recomendaciones de modelo

| Modelo     | RAM necesaria | Calidad   | Velocidad |
|------------|---------------|-----------|-----------|
| `mistral`  | ~6GB          | ★★★★☆    | Media     |
| `llama3`   | ~6GB          | ★★★★☆    | Media     |
| `gemma:2b` | ~3GB          | ★★☆☆☆    | Rápida    |

> **Nota**: `gemma:2b` tiene dificultades para seguir instrucciones de formato con prompts complejos. Se recomienda usar `mistral` o `llama3` para obtener resultados de calidad.

---

## Notebooks de referencia

- https://www.kaggle.com/code/soniaahlawat/sentiment-analysis-amazon-review
- https://www.kaggle.com/code/derrelldsouza/imdb-sentiment-analysis-eda-ml-lstm-bert
