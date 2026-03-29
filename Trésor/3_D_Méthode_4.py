from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# LDA utilise des fréquences brutes (CountVectorizer)
cv = CountVectorizer(max_features=500, stop_words='english')
X_counts = cv.fit_transform(df_clean['Context'].drop_duplicates())

lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(X_counts)

# Affichage des thèmes LDA
words = cv.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    indices = topic.argsort()[-5:][::-1]
    print(f"Sujet LDA {i} : {[words[idx] for idx in indices]}")