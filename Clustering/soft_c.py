import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel


def soft_clustering_lda(df, num_topics, nombre_analisis):
    print(f"\n--- [SOFT] Topic Modeling (LDA): {nombre_analisis} ---")

    # Preparar el texto para Gensim (lista de listas de palabras)
    # Stopwords
    stop_words = {'the', 'to', 'and', 'a', 'i', 'is', 'it', 'for', 'of', 'this', 'you', 'in', 'that', 'on', 'with',
                  'my', 'but', 'are', 'not', 'have', 'so', 'just', 'its', 'be', 'they', 'like', 'as', 'app'}

    textos = []
    for comentario in df['content_limpio']:
        palabras = [word for word in str(comentario).split() if word not in stop_words and len(word) > 2]
        textos.append(palabras)

    # Crear Diccionario y Corpus (formato Gensim)
    id2word = corpora.Dictionary(textos)

    # Filtramos palabras que aparecen en menos de 2 documentos o en más del 80%
    id2word.filter_extremes(no_below=2, no_above=0.8)
    corpus = [id2word.doc2bow(texto) for texto in textos]

    # Entrenar el modelo LDA
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=42, passes=10)

    # Mostrar los tópicos detectados
    for idx, topic in lda_model.print_topics(-1):
        print(f"Tópico {idx}: {topic}")

    # Extraer el tópico dominante y su probabilidad (para el CSV)
    topico_dominante = []
    probabilidad = []

    for row in corpus:
        # lda_model[row] devuelve una lista de tuplas con (id_topico, probabilidad)
        # Lo ordenamos para coger el de mayor probabilidad
        topicos_ordenados = sorted(lda_model[row], key=lambda x: (x[1]), reverse=True)
        if len(topicos_ordenados) > 0:
            topico_dominante.append(topicos_ordenados[0][0])
            probabilidad.append(round(topicos_ordenados[0][1] * 100, 2))  # En porcentaje
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
        df = pd.read_csv(archivo).dropna(subset=['content_limpio', 'sentimiento'])

        df_pos = df[df['sentimiento'] == 'Positivo'].copy()
        df_neg = df[df['sentimiento'] == 'Negativo'].copy()

        # Aplicamos Soft Clustering (K=4 para negativos, K=3 para positivos)
        df_neg_soft = soft_clustering_lda(df_neg, num_topics=4, nombre_analisis=f"{nombre_app} - Negativos")
        df_pos_soft = soft_clustering_lda(df_pos, num_topics=3, nombre_analisis=f"{nombre_app} - Positivos")

        # Guardamos resultados
        df_neg_soft.to_csv(f'{nombre_app}_Negativos_Soft.csv', index=False)
        df_pos_soft.to_csv(f'{nombre_app}_Positivos_Soft.csv', index=False)