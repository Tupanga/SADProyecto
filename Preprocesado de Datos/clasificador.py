import json
import pandas as pd
import joblib
import re
import os

def limpiar_texto(texto):
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^a-záéíóúñü\s0-9]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def ejecutar_clasificador(ruta_config):
    if not os.path.exists(ruta_config):
        print(f"Error: No se encuentra {ruta_config}")
        return

    with open(ruta_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    try:
        modelo = joblib.load(config["rutas"]["modelo_entrenado"])
        vectorizador = joblib.load(config["rutas"]["vectorizador"])
        df = pd.read_csv(config["rutas"]["archivo_entrada"])
    except Exception as e:
        print(f"Error al cargar archivos: {e}")
        return

    col = config["parametros_csv"]["columna_texto"]
    df['texto_limpio'] = df[col].apply(limpiar_texto)
    df_valido = df[df['texto_limpio'] != ''].copy()
    
    X = vectorizador.transform(df_valido['texto_limpio'])
    df_valido['prediccion'] = modelo.predict(X)
    
    print("\n--- Resultados ---")
    print(df_valido['prediccion'].value_counts())
    df_valido[[col, 'prediccion']].to_csv('resultados_finales.csv', index=False)
    print("\nArchivo 'resultados_finales.csv' generado.")

if __name__ == "__main__":
    ejecutar_clasificador('config.json')
