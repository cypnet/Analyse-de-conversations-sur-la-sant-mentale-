from sklearn.decomposition import TruncatedSVD

# On réduit la dimension de la matrice TF-IDF pour capturer les relations latentes
svd = TruncatedSVD(n_components=50)
X_embedded = svd.fit_transform(X)

kmeans_emb = KMeans(n_clusters=3, random_state=42)
kmeans_emb.fit(X_embedded)

# Extraction des mots clés sémantiques
for i in range(3):
    topic_weights = svd.components_.T @ kmeans_emb.cluster_centers_[i]
    indices = topic_weights.argsort()[-5:][::-1]
    print(f"Cluster {i} (Embeddings) : {[terms[idx] for idx in indices]}")