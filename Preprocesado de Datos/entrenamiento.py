import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import warnings

# Ocultar advertencias para que la consola se vea limpia
warnings.filterwarnings('ignore')

def limpiar_texto(texto):
    """Aplica las expresiones regulares para limpiar el texto base."""
    if not isinstance(texto, str): return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^a-záéíóúñü\s0-9]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def preprocesar_datos(df):
    """Transforma los datos crudos en datos listos para ML."""
    # 1. Eliminar columnas basura si existen
    columnas_basura = [col for col in df.columns if 'Unnamed' in col]
    df = df.drop(columns=columnas_basura)
    
    # 2. Eliminar filas sin texto
    df = df.dropna(subset=['content'])
    
    # 3. Castear score a número y eliminar errores de formato
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])
    
    # 4. Generar etiqueta (Target)
    def clasificar_sentimiento(score):
        if score <= 2: return 'Negativo'
        elif score == 3: return 'Neutro'
        else: return 'Positivo'
        
    df['sentimiento'] = df['score'].apply(clasificar_sentimiento)
    
    # 5. Limpieza NLP
    df['content_limpio'] = df['content'].apply(limpiar_texto)
    df = df[df['content_limpio'] != '']
    
    return df

def ejecutar_pipeline_entrenamiento(archivo_entrada, prefijo_salida):
    print(f"\n{'='*50}")
    print(f"🚀 INICIANDO PIPELINE DE ENTRENAMIENTO: {archivo_entrada}")
    print(f"{'='*50}")
    
    # --- FASE 1: PREPROCESADO ---
    print("1. Cargando y preprocesando datos originales...")
    try:
        df_crudo = pd.read_csv(archivo_entrada)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {archivo_entrada}")
        return
        
    df_limpio = preprocesar_datos(df_crudo)
    X_texto = df_limpio['content_limpio']
    y = df_limpio['sentimiento']
    
    # --- FASE 2: NLP Y VECTORIZACIÓN ---
    print("2. Vectorizando (TF-IDF, Unigramas + Bigramas, 7000 features)...")
    palabras_extra = ['app', 'application', 'hinge', 'boo', 'dating', 'just', 'like', 'really']
    mis_stopwords = list(ENGLISH_STOP_WORDS.union(palabras_extra))
    
    vectorizador = TfidfVectorizer(
        stop_words=mis_stopwords, 
        max_features=7000, 
        ngram_range=(1, 2)
    )
    X_numerico = vectorizador.fit_transform(X_texto)
    
    # --- FASE 3: SPLIT Y BALANCEO ---
    X_train, X_test, y_train, y_test = train_test_split(X_numerico, y, test_size=0.2, random_state=42)
    
    print("3. Balanceando los datos de entrenamiento con SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # --- FASE 4: ENTRENAMIENTO ---
    print("4. Entrenando modelo de Regresión Logística...")
    modelo = LogisticRegression(max_iter=1000, random_state=42)
    modelo.fit(X_train_bal, y_train_bal)
    
    # --- FASE 5: EVALUACIÓN ---
    print("5. Evaluando modelo final...")
    predicciones = modelo.predict(X_test)
    macro_f1 = f1_score(y_test, predicciones, average='macro')
    
    print(f"\n📊 Resultados para {archivo_entrada}:")
    print(f"--> MACRO F-SCORE: {macro_f1:.4f}")
    print(classification_report(y_test, predicciones))
    
    # --- FASE 6: EXPORTACIÓN ---
    nombre_modelo = f"modelo_entrenado_{prefijo_salida}.pkl"
    nombre_vectorizador = f"vectorizador_{prefijo_salida}.pkl"
    
    joblib.dump(modelo, nombre_modelo)
    joblib.dump(vectorizador, nombre_vectorizador)
    print(f"✅ Archivos exportados con éxito: {nombre_modelo} | {nombre_vectorizador}")

if __name__ == "__main__":
    # Asegúrate de tener Hinge.csv y Boo.csv en la misma carpeta que este script
    ejecutar_pipeline_entrenamiento('Hinge.csv', 'hinge')
    ejecutar_pipeline_entrenamiento('Boo.csv', 'boo')
    print("\n🎉 Pipeline de entrenamiento finalizado.")
