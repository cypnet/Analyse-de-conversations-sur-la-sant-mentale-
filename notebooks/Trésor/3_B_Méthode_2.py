from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Vectorisation
vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df_clean['Context'].drop_duplicates())

# Clustering en 3 groupes
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Affichage des mots dominants par cluster
terms = vectorizer.get_feature_names_out()
for i in range(3):
    indices = kmeans.cluster_centers_[i].argsort()[-5:][::-1]
    print(f"Cluster {i} (TF-IDF) : {[terms[idx] for idx in indices]}")