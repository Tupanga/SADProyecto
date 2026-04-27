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
        "Classify the sentiment of the following dating app review with exactly one word: "
        "positive (the user had a good experience), negative (the user had a bad experience), "
        "or neutral (the user had a neither good nor bad experience).\n\n"
        f'Review: "{text}"\n\n'
        "Answer (one word only):"
    )


def build_one_shot(text):
    return (
        "Classify the sentiment of the following dating app review as positive, negative, or neutral.\n\n"
        "Example:\n"
        'Review: "I havent matched with anyone yet, but it seems better than other apps so far"\n'
        "Answer: neutral\n\n"
        f'Review: "{text}"\n'
        "Answer (one word only):"
    )


def build_few_shot(text):
    return (
        "Classify the sentiment of the following dating app review as positive, negative, or neutral.\n\n"
        'Review: "Its very different from other dating apps. Just started, and Im more excited than my past experiences"\nAnswer: positive\n'
        'Review: "I dont want to date women young enough to be my daughter or granddaughter"\nAnswer: negative\n'
        'Review: "I havent matched with anyone yet, but it seems better than other apps so far."\nAnswer: neutral\n'
        'Review: "Good."\nAnswer: positive\n'
        'Review: "Great if youre looking for prostitutes"\nAnswer: negative\n'
        'Review: "Its alright"\nAnswer: neutral\n'
        'Review: "Idk if I can filter out the girls but I just want dudes"\nAnswer: neutral\n'
        'Review: "You advertise 50% off membership sale. Sounds great. I goto sign up. All the prices are the same as they have been except you have a number double the price crossed out... Thats actually bait and switch and not legal in most states."\nAnswer: negative\n'
        'Review: "Love the universe where we can interact without necessarily matching! Its like social media and dating app in one."\nAnswer: positive\n'
        f'Review: "{text}"\nAnswer (one word only):'
    )


def build_chain_of_thought(text):
    return (
        "You are a sentiment analysis expert. Analyze the following dating app review step by step and give a final label.\n\n"
        f'Review: "{text}"\n\n'
        "Step 1 - Identify emotional or meaningful words:\n"
        "Step 2 - Determine the overall tone:\n"
        "Final sentiment (positive / negative / neutral):"
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
# MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────
def confusion_matrix(predictions, gold):
    """Construye la matriz de confusión como dict {real: {predicho: count}}."""
    matrix = {label: {l: 0 for l in LABELS} for label in LABELS}
    for pred, true in zip(predictions, gold):
        if true in matrix and pred in matrix:
            matrix[true][pred] += 1
    return matrix


def print_confusion_matrix(matrix, strategy):
    """Muestra la matriz de confusión en la terminal de forma visual."""
    col_w = 12
    print(f"\n  {'─'*55}")
    print(f"  Matriz de Confusión  —  estrategia: {strategy}")
    print(f"  {'─'*55}")
    header = f"  {'Real vs Predicho':<16}" + "".join(f"{l.upper():>{col_w}}" for l in LABELS)
    print(header)
    print(f"  {'─'*55}")
    for true_label in LABELS:
        row = f"  {true_label.upper():<16}"
        for pred_label in LABELS:
            val = matrix[true_label][pred_label]
            # Resaltar diagonal (aciertos) con corchetes
            if true_label == pred_label:
                row += f"  [{'→'+str(val)+'←':>{col_w-4}}]  "
            else:
                row += f"{val:>{col_w}}"
        print(row)
    print(f"  {'─'*55}")
    print(f"  (Los valores entre [ ] son los aciertos por clase)")
    print()


def save_confusion_matrix_csv(matrix, strategy, path):
    """Guarda la matriz de confusión como CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["real_vs_predicho"] + [l.upper() for l in LABELS])
        for true_label in LABELS:
            row = [true_label.upper()] + [matrix[true_label][l] for l in LABELS]
            writer.writerow(row)
    print(f"  Guardado: {path}")


# ─────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────
def classify_dataset(rows, text_col, strategy, score_col=None, limit=None, debug=False):
    build_prompt = PROMPT_STRATEGIES[strategy]
    results = []
    subset = rows[:limit] if limit else rows
    total = len(subset)
    print(f"\n  Clasificando con '{strategy}' ({total} ejemplos)...")
    for i, row in enumerate(subset, 1):
        text           = row.get(text_col, "")
        original_label = score_to_sentiment(row.get(score_col, "")) if score_col else ""
        prompt = build_prompt(text)
        raw    = call_ollama(prompt)
        label  = parse_label(raw)
        if debug:
            print(f"\n  [{i}] TEXTO   : {text[:80]}")
            print(f"       RESPUESTA: {raw[:80]}")
            print(f"       ETIQUETA : {original_label} -> PREDICCION: {label}")
        results.append({
            "strategy":          strategy,
            "text":              text,
            "etiqueta_original": original_label,
            "prediccion":        label,
            "acierto":           "SI" if original_label == label else "NO",
            "raw_response":      raw,
        })
        if not debug and (i % 5 == 0 or i == total):
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

    strategies_to_run = (
        [args.strategy] if args.strategy else list(PROMPT_STRATEGIES.keys())
    )
    print(f"  Estrategias a evaluar: {strategies_to_run}")

    for strategy in strategies_to_run:
        results = classify_dataset(dev_rows, text_col, strategy, score_col=score_col, limit=args.limit, debug=args.debug)

        if score_col:
            gold    = [score_to_sentiment(r.get(score_col, "")) for r in dev_rows[:len(results)]]
            preds   = [r["prediccion"] for r in results]
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

    # Generar matriz de confusión de la mejor estrategia
    if score_col and best_results:
        gold_best  = [score_to_sentiment(r.get(score_col, "")) for r in dev_rows[:len(best_results)]]
        preds_best = [r["prediccion"] for r in best_results]
        cm = confusion_matrix(preds_best, gold_best)
        print_confusion_matrix(cm, best_strategy)
        # Se guardará en el paso 4
    else:
        cm = None

    # ── 3. Oversampling generativo ───────────────────────────
    print("\n[3/4] Oversampling generativo (clase minoritaria)...")
    if args.skip_oversampling:
        print("  Oversampling desactivado (--skip_oversampling).")
        generated = []
    elif score_col:
        generated = oversample_minority(train_rows, text_col, score_col,
                                        minority_label, n_per_sample=args.n_paraphrases)
    else:
        print("  Sin columna de etiqueta, saltando oversampling.")
        generated = []

    # ── 4. Guardar resultados ────────────────────────────────
    print("\n[4/4] Guardando resultados...")
    best_clean = [
        {
            "text":              r["text"],
            "etiqueta_original": r["etiqueta_original"],
            "prediccion":        r["prediccion"],
            "acierto":           r["acierto"],
        }
        for r in best_results
    ]
    save_predictions(best_clean, str(out_dir / "predictions_best.csv"))
    save_prompts_log(prompt_log,  str(out_dir / "prompts_log.csv"))
    if cm:
        save_confusion_matrix_csv(cm, best_strategy, str(out_dir / "confusion_matrix.csv"))
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
    parser.add_argument("--strategy", default=None,
                        choices=["zero_shot", "one_shot", "few_shot", "chain_of_thought"],
                        help="Estrategia de prompting a usar (por defecto prueba las 4)")
    parser.add_argument("--skip_oversampling", action="store_true",
                        help="Saltar el oversampling generativo")
    parser.add_argument("--debug", action="store_true",
                        help="Mostrar respuesta raw del modelo para cada reseña")
    args = parser.parse_args()
    main(args)
