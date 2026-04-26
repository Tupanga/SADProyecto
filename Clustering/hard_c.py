import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def hard_clustering(df, n_clusters, nombre_analisis):
    print(f"\n--- [HARD] Clustering: {nombre_analisis} ---")

    # Vectorización
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['content_limpio'])

    # Entrenamiento K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Al ser "Hard": se le asigna UN solo clúster 
    df.loc[:, 'cluster_hard'] = kmeans.fit_predict(X_tfidf)

    # Extraer top palabras
    diccionario = vectorizer.get_feature_names_out()
    centroides = kmeans.cluster_centers_
    for i, centroide in enumerate(centroides):
        top_indices = centroide.argsort()[-10:][::-1]
        top_palabras = [diccionario[idx] for idx in top_indices]
        print(f"Clúster {i}: {', '.join(top_palabras)}")

    return df


if __name__ == "__main__":
    archivos = ['Hinge_preprocesado.csv', 'Boo_preprocesado.csv']

    for archivo in archivos:
        nombre_app = archivo.split('_')[0]
        df = pd.read_csv(archivo).dropna(subset=['content_limpio', 'sentimiento'])

        df_pos = df[df['sentimiento'] == 'Positivo'].copy()
        df_neg = df[df['sentimiento'] == 'Negativo'].copy()

        # Aplicamos Hard Clustering (K=4 para negativos, K=3 para positivos)
        df_neg_hard = hard_clustering(df_neg, n_clusters=4, nombre_analisis=f"{nombre_app} - Negativos")
        df_pos_hard = hard_clustering(df_pos, n_clusters=3, nombre_analisis=f"{nombre_app} - Positivos")

        # Guardamos resultados
        df_neg_hard.to_csv(f'{nombre_app}_Negativos_Hard.csv', index=False)
        df_pos_hard.to_csv(f'{nombre_app}_Positivos_Hard.csv', index=False)