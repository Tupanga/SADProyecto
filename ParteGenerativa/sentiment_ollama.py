"""
Parte Generativa - Sentiment Analysis con Ollama
Proyecto: Toma de decisiones - ¿Podemos mejorar las valoraciones de nuestra aplicación?

Estrategias implementadas:
  - Zero-shot classification
  - One-shot classification
  - Few-shot classification
  - Oversampling generativo (paráfrasis de clase minoritaria)

Salidas:
  - predictions_best.csv  -> texto + predicción del mejor prompt
  - prompts_log.csv       -> todos los prompts probados con ejemplos E/S
  - paraphrases.csv       -> paráfrasis generadas para oversampling
  - train_split.csv       -> 80% de los datos (generado automáticamente)
  - dev_split.csv         -> 20% de los datos (generado automáticamente)
"""

import csv
import json
import time
import random
import argparse
import requests
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL       = "gemma:2b"
LABELS      = ["positive", "negative", "neutral"]
LABEL_ES    = {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}
MAX_RETRIES = 3

# ─────────────────────────────────────────────
# LLAMADA A OLLAMA
# ─────────────────────────────────────────────
def call_ollama(prompt, model=None, temperature=0.0):
    if model is None:
        model = MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 50},
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            print(f"  [Intento {attempt+1}/{MAX_RETRIES}] Error: {e}")
            time.sleep(2)
    return ""


def parse_label(raw):
    """
    Extrae el sentimiento de la respuesta del modelo.
    Acepta tanto español (Positivo/Negativo/Neutral) como inglés (positive/negative/neutral).
    """
    raw_lower = raw.lower().strip()
    # Buscar primero la primera palabra de la respuesta (el modelo debe responder una sola palabra)
    first_word = raw_lower.split()[0].rstrip(".,;:") if raw_lower.split() else ""

    # Español
    if first_word in ("positivo", "positive"):  return "positive"
    if first_word in ("negativo", "negative"):  return "negative"
    if first_word in ("neutral", "neutro"):     return "neutral"

    # Buscar en toda la respuesta como fallback
    if "positivo" in raw_lower or "positive" in raw_lower: return "positive"
    if "negativo" in raw_lower or "negative" in raw_lower: return "negative"
    if "neutral"  in raw_lower or "neutro"   in raw_lower: return "neutral"

    return "unknown"


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────


def build_zero_shot(text):
    return (
        "Clasifica los sentimientos que el usuario ha querido retransmitir en la siguiente reseña de una pagina"
        " de citas con una de estas palabras:"
        " Positivo(Cuando la experiencia de usuario a sido agradable), Negativo(Cuando la experiencia de usuario a sido desagradable), or Neutral(Cuando la experiencia de usuario no sido ni agradable ni desagradable).\n\n"
        f'Reseña: "{text}"\n\n'
        "Respuesta (Una unica palabra):"
    )


def build_one_shot(text):
    return (
        "Clasifica los sentimientos de la siguiente reseña de una pagina de citas con las siguientes opciones: Positivo, Negativo, or Neutral.\n\n"
        "Ejemplo:\n"
        'Reseña: "I havent matched with anyone yet, but it seems better than other apps so far"\n'
        "Respuesta: Neutral\n\n"
        f'Reseña: "{text}"\n'
        "Respuesta (Una unica palabra):"
    )


def build_few_shot(text):
    return (
        "Clasifica la siguiente reseña de una pagina de citas con las siguientes opciones: Positivo, Negativo, or Neutral.\n\n"
        'Reseña: "Its very different from other dating apps. Just started, and Im more excited than my past experiences"\nRespuesta: Positivo\n'
        'Reseña: "I dont want to date women young enough to be my daughter or granddaughter"\nRespuesta: Negativo\n'
        'Reseña: "I havent matched with anyone yet, but it seems better than other apps so far."\nRespuesta: Neutral\n'
        'Reseña: "Good."\nRespuesta: Positivo\n'
        'Reseña: "aGreat if youre looking for prostitutes"\nRespuesta: Negativo\n'
        'Reseña: "Its alright"\nRespuesta: Neutral\n'
        'Reseña: "Its alr Idk if I can filter out the girls but I just want dudes"\nRespuesta: Neutral\n'
        'Reseña: "You advertise 50% off membership sale. Sounds great. I goto sign up. All the prices are the same as they have been except you have a number double the price crosssed out... Thats actually bait and switch and not legal in most states knock the price to half on ALL your price tiers... Really half off not fake half off... I would try. Was thinking about trying at regular orice but not after you pulling this. Smh"\nRespuesta: Negativo\n'
        'Reseña: "Love the universe where we can interact without necessarily matching! Its like social media and dating/friendship app in on."\nRespuesta: Positivo\n'
        f'Reseña: "{text}"\nRespuesta (Una unica palabra):'
    )


def build_chain_of_thought(text):
    return (
        "Eres un analista de sentimientos experto. Tu labor es realizar un analisis de sentimientos paso por paso de una reseña de una pagina de citas y entregar una sentencia final.\n\n"
        f'Reseña: "{text}"\n\n'
        "Paso 1 - Identifica las palabras emocionales o que tienen un significado emocional:\n"
        "Paso 2 - Determinar el tono general de la frase:\n"
        "Sentimiento final (Positivo / Negativo / Neutral):"
    )


PROMPT_STRATEGIES = {
    "zero_shot":        build_zero_shot,
    "one_shot":         build_one_shot,
    "few_shot":         build_few_shot,
    "chain_of_thought": build_chain_of_thought,
}


# ─────────────────────────────────────────────
# LECTURA Y DIVISIÓN DE DATOS
# ─────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"  Cargadas {len(rows)} filas de '{path}'")
    return rows


def score_to_sentiment(score):
    v = str(score).strip().lower()
    if v in ("positivo", "positive"): return "positive"
    if v in ("negativo", "negative"): return "negative"
    if v in ("neutro",   "neutral"):  return "neutral"
    try:
        s = float(v)
        if s >= 4:  return "positive"
        if s <= 2:  return "negative"
        return "neutral"
    except ValueError:
        return "unknown"


def split_dataset(rows, train_ratio=0.8, seed=42):
    """Split estratificado por clase."""
    random.seed(seed)
    by_class = {}
    for row in rows:
        label = score_to_sentiment(row.get("sentimiento") or row.get("score", ""))
        by_class.setdefault(label, []).append(row)

    train, dev = [], []
    for label, items in by_class.items():
        random.shuffle(items)
        cut = max(1, int(len(items) * train_ratio))
        train.extend(items[:cut])
        dev.extend(items[cut:])

    random.shuffle(train)
    random.shuffle(dev)
    return train, dev


def detect_columns(rows):
    keys = list(rows[0].keys()) if rows else []
    if "content_limpio" in keys:
        text_col = "content_limpio"
    elif "content" in keys:
        text_col = "content"
    else:
        text_col = next((k for k in keys if "text" in k.lower() or "review" in k.lower()), keys[0])

    if "sentimiento" in keys:
        score_col = "sentimiento"
    elif "score" in keys:
        score_col = "score"
    else:
        score_col = next((k for k in keys if "label" in k.lower() or "rating" in k.lower()), None)

    return text_col, score_col


def save_split(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Guardado split: {path}  ({len(rows)} filas)")


# ─────────────────────────────────────────────
# EVALUACIÓN
# ─────────────────────────────────────────────
def evaluate(predictions, gold):
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for pred, true in zip(predictions, gold):
        if pred == true:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    metrics = {}
    for label in LABELS:
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        rec  = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        metrics[label] = {"precision": prec, "recall": rec, "f1": f1}
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(metrics)
    correct  = sum(p == g for p, g in zip(predictions, gold))
    metrics["macro_f1"] = macro_f1
    metrics["accuracy"] = correct / len(gold) if gold else 0
    return metrics


# ─────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────
def classify_dataset(rows, text_col, strategy, limit=None):
    build_prompt = PROMPT_STRATEGIES[strategy]
    results = []
    subset = rows[:limit] if limit else rows
    total = len(subset)
    print(f"\n  Clasificando con '{strategy}' ({total} ejemplos)...")
    for i, row in enumerate(subset, 1):
        text   = row.get(text_col, "")
        prompt = build_prompt(text)
        raw    = call_ollama(prompt)
        label  = parse_label(raw)
        results.append({
            "strategy":     strategy,
            "text":         text,
            "raw_response": raw,
            "prediction":   label,
        })
        if i % 5 == 0 or i == total:
            print(f"    {i}/{total} procesados", end="\r")
    print()
    return results


# ─────────────────────────────────────────────
# OVERSAMPLING GENERATIVO
# ─────────────────────────────────────────────
PARAPHRASE_PROMPT = (
    "Generate {n} diverse paraphrases of the following review that preserve "
    "its original sentiment. Return ONLY a JSON array of strings, no extra text.\n\n"
    'Original review: "{text}"\n\nJSON array:'
)

def generate_paraphrases(text, n=3):
    prompt = PARAPHRASE_PROMPT.format(text=text, n=n)
    raw = call_ollama(prompt, temperature=0.8)
    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass
    lines = [l.strip().strip('"').strip("'").strip("-").strip() for l in raw.splitlines()]
    return [l for l in lines if len(l) > 10][:n]


def oversample_minority(rows, text_col, score_col, target_label, n_per_sample=2):
    minority = [r for r in rows if score_to_sentiment(r.get(score_col, "")) == target_label]
    print(f"\n  Oversampling '{target_label}': {len(minority)} ejemplos originales")
    new_rows = []
    for i, row in enumerate(minority, 1):
        text  = row.get(text_col, "")
        paras = generate_paraphrases(text, n=n_per_sample)
        for p in paras:
            new_row = dict(row)
            new_row[text_col]     = p
            new_row["_generated"] = "true"
            new_rows.append(new_row)
        print(f"    {i}/{len(minority)} procesados", end="\r")
    print(f"\n  Generadas {len(new_rows)} muestras nuevas para '{target_label}'")
    return new_rows


# ─────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────
def save_predictions(results, path):
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Guardado: {path}")


def save_prompts_log(log, path):
    if not log:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "model", "strategy", "macro_f1",
                                               "accuracy", "prompt_example",
                                               "input_example", "output_example"])
        writer.writeheader()
        writer.writerows(log)
    print(f"  Guardado: {path}")


def save_paraphrases(rows_original, rows_generated, text_col, score_col, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "sentiment", "is_generated"])
        for r in rows_original:
            writer.writerow([r.get(text_col, ""),
                             score_to_sentiment(r.get(score_col, "")), "false"])
        for r in rows_generated:
            writer.writerow([r.get(text_col, ""),
                             score_to_sentiment(r.get(score_col, "")), "true"])
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# FLUJO PRINCIPAL
# ─────────────────────────────────────────────
def main(args):
    global MODEL
    MODEL = args.model

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  PARTE GENERATIVA - Sentiment Analysis con Ollama")
    print(f"  Modelo: {MODEL}")
    print("="*60)

    # ── 1. Cargar y dividir ──────────────────────────────────
    print("\n[1/4] Cargando y dividiendo datos (80% train / 20% dev)...")
    all_rows = load_csv(args.input)
    text_col, score_col = detect_columns(all_rows)
    print(f"  Columna texto    : '{text_col}'")
    print(f"  Columna etiqueta : '{score_col}'")

    train_rows, dev_rows = split_dataset(all_rows, train_ratio=0.8)
    print(f"  Split → train: {len(train_rows)} filas | dev: {len(dev_rows)} filas")

    save_split(train_rows, str(out_dir / "train_split.csv"))
    save_split(dev_rows,   str(out_dir / "dev_split.csv"))

    if score_col:
        dist = Counter(score_to_sentiment(r.get(score_col, "")) for r in train_rows)
        print(f"  Distribución train: {dict(dist)}")
        minority_label = min(dist, key=dist.get)
        print(f"  Clase minoritaria : '{minority_label}'")
    else:
        minority_label = "neutral"

    # ── 2. Evaluar estrategias ───────────────────────────────
    print("\n[2/4] Evaluando estrategias de prompting en dev...")
    prompt_log = []
    best_strategy, best_f1, best_results = None, -1, []

    for strategy in PROMPT_STRATEGIES:
        results = classify_dataset(dev_rows, text_col, strategy, limit=args.limit)

        if score_col:
            gold    = [score_to_sentiment(r.get(score_col, "")) for r in dev_rows[:len(results)]]
            preds   = [r["prediction"] for r in results]
            metrics = evaluate(preds, gold)
            mf1     = metrics["macro_f1"]
            acc     = metrics["accuracy"]
            print(f"  {strategy:20s}  Macro-F1={mf1:.3f}  Acc={acc:.3f}")
        else:
            mf1, acc = 0, 0

        ex = results[0] if results else {}
        prompt_log.append({
            "rank":           "",
            "model":          MODEL,
            "strategy":       strategy,
            "macro_f1":       f"{mf1:.4f}",
            "accuracy":       f"{acc:.4f}",
            "prompt_example": PROMPT_STRATEGIES[strategy]("[TEXTO DE EJEMPLO]"),
            "input_example":  ex.get("text", "")[:200],
            "output_example": ex.get("raw_response", "")[:200],
        })

        if mf1 > best_f1:
            best_f1, best_strategy, best_results = mf1, strategy, results

    prompt_log.sort(key=lambda x: float(x["macro_f1"]), reverse=True)
    for i, row in enumerate(prompt_log, 1):
        row["rank"] = i

    print(f"\n  ✓ Mejor estrategia: '{best_strategy}' (Macro-F1={best_f1:.3f})")

    # ── 3. Oversampling generativo ───────────────────────────
    print("\n[3/4] Oversampling generativo (clase minoritaria)...")
    if score_col:
        generated = oversample_minority(train_rows, text_col, score_col,
                                        minority_label, n_per_sample=args.n_paraphrases)
    else:
        print("  Sin columna de etiqueta, saltando oversampling.")
        generated = []

    # ── 4. Guardar resultados ────────────────────────────────
    print("\n[4/4] Guardando resultados...")
    best_clean = [{"text": r["text"], "prediction": r["prediction"]} for r in best_results]
    save_predictions(best_clean, str(out_dir / "predictions_best.csv"))
    save_prompts_log(prompt_log,  str(out_dir / "prompts_log.csv"))
    if score_col:
        save_paraphrases(train_rows, generated, text_col, score_col,
                         str(out_dir / "paraphrases.csv"))

    summary = {
        "model":               MODEL,
        "input_file":          args.input,
        "total_rows":          len(all_rows),
        "train_rows":          len(train_rows),
        "dev_rows":            len(dev_rows),
        "best_strategy":       best_strategy,
        "best_macro_f1":       round(best_f1, 4),
        "minority_label":      minority_label,
        "generated_samples":   len(generated),
        "strategies_evaluated": list(PROMPT_STRATEGIES.keys()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Guardado: {out_dir / 'summary.json'}")

    print("\n" + "="*60)
    print("  ✓ Proceso completado")
    print(f"  Resultados en: '{out_dir}'")
    print("="*60 + "\n")


# ─────────────────────────────────────────────
# ARGUMENTOS CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis generativo con Ollama"
    )
    parser.add_argument("--input",         default="Hinge_preprocesado.csv",
                        help="CSV preprocesado (único fichero de entrada)")
    parser.add_argument("--output_dir",    default="output",
                        help="Carpeta de salida")
    parser.add_argument("--model",         default=MODEL,
                        help="Modelo Ollama (gemma:2b, llama3, mistral...)")
    parser.add_argument("--limit",         type=int, default=None,
                        help="Nº máximo de ejemplos de dev a evaluar (None = todos)")
    parser.add_argument("--n_paraphrases", type=int, default=2,
                        help="Nº de paráfrasis por muestra en oversampling")
    args = parser.parse_args()
    main(args)