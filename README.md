# Parte Generativa — Sentiment Analysis con Ollama

## Requisitos previos

1. Tener [Ollama](https://ollama.com/) instalado y corriendo:
   ```bash
   ollama serve
   ollama pull llama3      # o el modelo que prefieras
   ```

2. Instalar dependencias Python:
   ```bash
   pip install -r requirements.txt
   ```

---

## Uso básico

```bash
python sentiment_ollama.py --train train.csv --dev dev.csv
```

### Opciones

| Argumento         | Por defecto | Descripción                                       |
|-------------------|-------------|---------------------------------------------------|
| `--train`         | `train.csv` | Fichero de entrenamiento                          |
| `--dev`           | `dev.csv`   | Fichero de validación                             |
| `--output_dir`    | `output/`   | Carpeta donde se guardan los resultados           |
| `--model`         | `llama3`    | Modelo de Ollama (llama3, mistral, phi3…)         |
| `--limit`         | `None`      | Limitar nº de ejemplos de dev (útil para pruebas)|
| `--n_paraphrases` | `2`         | Paráfrasis por muestra en oversampling generativo |

### Ejemplo completo

```bash
python sentiment_ollama.py \
  --train train.csv \
  --dev dev.csv \
  --model llama3 \
  --limit 50 \
  --n_paraphrases 3 \
  --output_dir resultados/
```

---

## Ficheros de salida

| Fichero                    | Contenido                                                        |
|----------------------------|------------------------------------------------------------------|
| `predictions_best.csv`     | Texto + predicción con el mejor prompt encontrado               |
| `prompts_log.csv`          | Todos los prompts probados, métricas, ejemplos entrada/salida   |
| `paraphrases.csv`          | Dataset aumentado (originales + generadas para oversampling)    |
| `summary.json`             | Resumen: modelo, mejor estrategia, Macro-F1, nº muestras nuevas|

---

## Estructura esperada del CSV de entrada

El script detecta automáticamente las columnas, pero idealmente:

```
text,score
"The app is amazing, I love it",5
"Terrible experience, crashes all the time",1
"It works fine, nothing special",3
```

- **Columna de texto**: cualquier columna con `text`, `review` o `comment` en el nombre.
- **Columna de score**: cualquier columna con `score`, `label`, `rating` o `sentiment` en el nombre.
- El score numérico se convierte: `>=4 → positive`, `<=2 → negative`, `3 → neutral`.

---

## Estrategias de prompting implementadas

| Estrategia          | Descripción                                           |
|---------------------|-------------------------------------------------------|
| `zero_shot`         | Sin ejemplos, solo instrucción directa               |
| `one_shot`          | Un ejemplo de referencia                             |
| `few_shot`          | Cinco ejemplos variados                              |
| `chain_of_thought`  | Razonamiento paso a paso antes de la predicción      |

El script las evalúa todas en `dev.csv` y elige la de mayor **Macro-F1**.

---

## Notebooks de referencia

- https://www.kaggle.com/code/soniaahlawat/sentiment-analysis-amazon-review
- https://www.kaggle.com/code/derrelldsouza/imdb-sentiment-analysis-eda-ml-lstm-bert
