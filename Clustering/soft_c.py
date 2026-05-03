import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


def evaluar_coherencia_lda(textos, corpus, id2word, max_topics, titulo):
    """Calcula y grafica el Coherence Score para encontrar el K óptimo en LDA."""
    print(f"Calculando métricas de coherencia para {titulo}...")
    coherence_values = []
    rango_k = range(2, max_topics + 1)

    for k in rango_k:
        # Entrenamos un modelo temporal para cada K
        modelo_lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=k, random_state=42, passes=5)
        # Calculamos su coherencia
        coherencemodel = CoherenceModel(model=modelo_lda, texts=textos, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    # Generamos el gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, coherence_values, marker='o', linestyle='-', color='g')
    plt.title(f'Coherencia de Tópicos (LDA) - {titulo}')
    plt.xlabel('Número de Tópicos (k)')
    plt.ylabel('Coherencia (C_v)')
    plt.grid(True)

    nombre_imagen = f'coherencia_{titulo.lower().replace(" ", "_")}.png'
    plt.savefig(nombre_imagen)
    print(f"Gráfico guardado como: {nombre_imagen}")
    plt.show()


def soft_clustering_lda(df, num_topics, nombre_analisis, evaluar_k):
    print(f"\n--- [SOFT] Topic Modeling (LDA): {nombre_analisis} ---")

    # Preparar el texto para Gensim
    stop_words = {'the', 'to', 'and', 'a', 'i', 'is', 'it', 'for', 'of', 'this', 'you', 'in', 'that', 'on', 'with',
                  'my', 'but', 'are', 'not', 'have', 'so', 'just', 'its', 'be', 'they', 'like', 'as', 'app', 'dont'}

    textos = []
    for comentario in df['content_limpio']:
        palabras = [word for word in str(comentario).split() if word not in stop_words and len(word) > 2]
        textos.append(palabras)

    # Crear Diccionario y Corpus
    id2word = corpora.Dictionary(textos)
    id2word.filter_extremes(no_below=2, no_above=0.8)
    corpus = [id2word.doc2bow(texto) for texto in textos]

    # Generar el gráfico de coherencia para decidir K
    if evaluar_k:
        evaluar_coherencia_lda(textos, corpus, id2word, max_topics=8, titulo=nombre_analisis)

    # Entrenar el modelo LDA definitivo
    print(f"Entrenando modelo final con K={num_topics}...")
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10)

    # Mostrar los tópicos detectados
    for idx, topic in lda_model.print_topics(-1):
        print(f"Tópico {idx}: {topic}")

    # Extraer el tópico dominante y su probabilidad
    topico_dominante = []
    probabilidad = []

    for row in corpus:
        topicos_ordenados = sorted(lda_model[row], key=lambda x: (x[1]), reverse=True)
        if len(topicos_ordenados) > 0:
            topico_dominante.append(topicos_ordenados[0][0])
            probabilidad.append(round(topicos_ordenados[0][1] * 100, 2))
        else:
            topico_dominante.append(-1)
            probabilidad.append(0)

    df.loc[:, 'topico_principal_soft'] = topico_dominante
    df.loc[:, 'porcentaje_pertenencia'] = probabilidad

    return df


if __name__ == "__main__":
    archivos = ['Hinge_preprocesado.csv', 'Boo_preprocesado.csv']

    for archivo in archivos:
        nombre_app = archivo.split('_')[0]
        # Usamos try/except por si el archivo no existe en el directorio local al copiar el código
        try:
            df = pd.read_csv(archivo).dropna(subset=['content_limpio', 'sentimiento'])

            df_pos = df[df['sentimiento'] == 'Positivo'].copy()
            df_neg = df[df['sentimiento'] == 'Negativo'].copy()

            # Aplicamos Soft Clustering.
            df_neg_soft = soft_clustering_lda(df_neg, num_topics=4, nombre_analisis=f"{nombre_app} - Negativos",
                                              evaluar_k=True)
            df_pos_soft = soft_clustering_lda(df_pos, num_topics=3, nombre_analisis=f"{nombre_app} - Positivos",
                                              evaluar_k=True)

            # Guardamos resultados
            df_neg_soft.to_csv(f'{nombre_app}_Negativos_Soft.csv', index=False)
            df_pos_soft.to_csv(f'{nombre_app}_Positivos_Soft.csv', index=False)

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {archivo}. Asegúrate de que está en la misma carpeta.")
