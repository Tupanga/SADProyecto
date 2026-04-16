import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def cargar_y_filtrar(ruta_archivo):

    """Carga el dataset preprocesado y devuelve dataframes separados por sentimiento."""
    print(f"Cargando datos de: {ruta_archivo}")
    df = pd.read_csv(ruta_archivo)

    # Filtramos eliminando valores nulos por seguridad
    df = df.dropna(subset=['content_limpio', 'sentimiento'])

    df_positivos = df[df['sentimiento'] == 'Positivo'].copy()
    df_negativos = df[df['sentimiento'] == 'Negativo'].copy()

    return df_positivos, df_negativos


def metodo_del_codo(X, max_k, titulo):
    """Genera y guarda el gráfico del codo para determinar el K óptimo."""
    inercia = []
    rango_k = range(1, max_k + 1)

    print(f"Calculando inercia para gráfico del codo ({titulo})...")
    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inercia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, inercia, marker='o', linestyle='-', color='b')
    plt.title(f'Método del Codo - {titulo}')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('Inercia')
    plt.grid(True)

    # Guardamos la imagen para el póster
    nombre_imagen = f'codo_{titulo.lower().replace(" ", "_")}.png'
    plt.savefig(nombre_imagen)
    print(f"Gráfico guardado como: {nombre_imagen}")
    plt.show()


def extraer_top_palabras(kmeans, vectorizer, n_palabras=10):
    """Extrae las palabras más significativas (Top N) de cada clúster."""
    diccionario_palabras = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_

    top_palabras_por_cluster = {}
    for i, centroide in enumerate(centroides):
        # Ordenamos los índices de los pesos de menor a mayor y tomamos los últimos (los mayores)
        top_indices = centroide.argsort()[-n_palabras:][::-1]
        top_palabras = [diccionario_palabras[idx] for idx in top_indices]
        top_palabras_por_cluster[f"Clúster {i}"] = top_palabras

    return top_palabras_por_cluster


def ejecutar_clustering(df, n_clusters, nombre_analisis):
    """Pipeline completo de clustering para un dataframe dado."""
    print(f"\n--- Iniciando Clustering: {nombre_analisis} ---")

    # Vectorización TF-IDF
    # Si los comentarios son en inglés,'english'. Si son en español,'spanish'.
    # max_features limita el vocabulario a las palabras más comunes para evitar ruido extremo
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X_tfidf = vectorizer.fit_transform(df['content_limpio'])

    #  Metodo del Codo
    metodo_del_codo(X_tfidf, max_k=10, titulo=nombre_analisis)

    #  Aplicar K-Means con el K seleccionado
    print(f"Entrenando K-Means con k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[:, 'cluster'] = kmeans.fit_predict(X_tfidf)

    #  Extraer las palabras representativas
    palabras_clave = extraer_top_palabras(kmeans, vectorizer, n_palabras=10)

    print("\nPalabras más significativas por Clúster:")
    for cluster, palabras in palabras_clave.items():
        print(f"{cluster}: {', '.join(palabras)}")

    return df, palabras_clave


if __name__ == "__main__":

    # Ponemos los dos archivos en una lista
    archivos_a_analizar = ['Hinge_preprocesado.csv', 'Boo_preprocesado.csv']

    # Número de clústeres por defecto
    K_BOO_P = 3
    K_BOO_N = 4
    K_HINGE_P = 3
    K_HINGE_N = 5
    K = 0


    for archivo in archivos_a_analizar:
        print("=" * 50)
        print(f" INICIANDO ANÁLISIS PARA: {archivo}")
        print("=" * 50)


        #  Cargar datos
        df_pos, df_neg = cargar_y_filtrar(archivo)
        print(f"Opiniones Positivas: {len(df_pos)} | Opiniones Negativas: {len(df_neg)}")

        # Extraemos el nombre de la app (Hinge o Boo) para los títulos
        nombre_app = archivo.split('_')[0]

        # 2. Analizar Negativos
        if archivo.title() == 'Hinge_preprocesado.csv':
            K = K_HINGE_N
        else:
            K = K_BOO_N

        df_neg_clusterizado, keywords_neg = ejecutar_clustering(
            df_neg,
            n_clusters=K,
            nombre_analisis=f"{nombre_app} - Comentarios Negativos"
        )

        # 3. Analizar Positivos

        if archivo.title() == 'Hinge_preprocesado.csv':
            K = K_HINGE_P
        else:
            K = K_BOO_P

        df_pos_clusterizado, keywords_pos = ejecutar_clustering(
            df_pos,
            n_clusters=K,
            nombre_analisis=f"{nombre_app} - Comentarios Positivos"
        )

        # 4. Guardar los resultados
        df_neg_clusterizado.to_csv(f'{nombre_app}_Negativos_Clusters.csv', index=False)
        df_pos_clusterizado.to_csv(f'{nombre_app}_Positivos_Clusters.csv', index=False)

        print(f"\n Análisis de {nombre_app} finalizado. Archivos guardados.\n")
