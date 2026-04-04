import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import re
import warnings

warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from gensim.models import Word2Vec

# Sélection du fichier
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()

print("Sélectionner le fichier train.csv")
chemin_fichier = filedialog.askopenfilename(
    title="Sélectionne ton fichier train.csv",
    filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
)

# Nettoyage des données
print("\n--- 1. NETTOYAGE DES DONNÉES")
df_brut = pd.read_csv(chemin_fichier)

df = df_brut.dropna().drop_duplicates().copy()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) 
    return text

df['Context_clean'] = df['Context'].apply(clean_text)
df['Response_clean'] = df['Response'].apply(clean_text)

print(f"Lignes conservées après nettoyage : {len(df)}")

# Statistiques globales
print("\n--- 2. STATISTIQUES RÉCAPITULATIVES")
nb_patients = df['Context'].nunique()
nb_therapists = df['Response'].nunique()
avg_exchanges = len(df) / nb_patients

dist_therapists = df.groupby('Response')['Context'].count()

print(f"Nombre de patients uniques                 : {nb_patients}")
print(f"Nombre de thérapeutes uniques              : {nb_therapists}")
print(f"Nombre d'échanges moyen par conversation   : {avg_exchanges:.2f}")
print(f"Le thérapeute le plus actif a répondu à    : {dist_therapists.max()} conversations différentes")

# Dictionnaire lexical
print("\n--- 3. DICTIONNAIRE LEXICAL")
vec_ngrams = CountVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))

def exporter_frequence(textes, nom_fichier):
    matrice = vec_ngrams.fit_transform(textes)
    frequences = zip(vec_ngrams.get_feature_names_out(), matrice.sum(axis=0).tolist()[0])
    df_freq = pd.DataFrame(frequences, columns=['Mot/Expression', 'Frequence']).sort_values(by='Frequence', ascending=False)
    df_freq.to_csv(nom_fichier, index=False)
    return df_freq

patients_uniques = df['Context_clean'].drop_duplicates()
export_pat = exporter_frequence(patients_uniques, 'stats_mots_patients.csv')

export_ther = exporter_frequence(df['Response_clean'], 'stats_mots_therapeutes.csv')

premiere_conv = [df.iloc[0]['Context_clean'] + " " + df.iloc[0]['Response_clean']]
export_conv = exporter_frequence(premiere_conv, 'stats_mots_premiere_conv.csv')

print("    Les 3 fichiers CSV ont été générés avec succès.")

# Extraction de sujets
print("\n--- 4. IDENTIFICATION DES SUJETS MAJEURS")

print("\n  Méthode 1 : Lexique manuel")
lexique = ['stress', 'anxiety', 'sleep', 'depression', 'family', 'trauma', 'suicide']
vec_lexique = CountVectorizer(vocabulary=lexique)
comptage_lexique = vec_lexique.fit_transform(patients_uniques).sum(axis=0).tolist()[0]
resultat_lexique = dict(zip(lexique, comptage_lexique))
print("Occurrences des mots choisis :", resultat_lexique)

print("\n  Méthode 2 : Clusters TF-IDF + K-Means")
tfidf = TfidfVectorizer(stop_words='english', max_features=500)
X_tfidf = tfidf.fit_transform(patients_uniques)
kmeans_mots = KMeans(n_clusters=4, random_state=42).fit(X_tfidf.T)
mots_tfidf = tfidf.get_feature_names_out()

for i in range(4):
    mots_du_cluster = [mots_tfidf[j] for j in range(len(mots_tfidf)) if kmeans_mots.labels_[j] == i][:5]
    print(f"Cluster TF-IDF {i+1} : {', '.join(mots_du_cluster)}")

print("\n  Méthode 3 : Clusters Word Embeddings (Word2Vec)")
phrases_coupees = [texte.split() for texte in patients_uniques]
w2v = Word2Vec(phrases_coupees, vector_size=50, window=5, min_count=5, workers=4)

mots_w2v = w2v.wv.index_to_key
vecteurs_w2v = w2v.wv.vectors
kmeans_w2v = KMeans(n_clusters=4, random_state=42).fit(vecteurs_w2v)

for i in range(4):
    mots_du_cluster_w2v = [mots_w2v[j] for j in range(len(mots_w2v)) if kmeans_w2v.labels_[j] == i][:5]
    print(f"Cluster Word2Vec {i+1} : {', '.join(mots_du_cluster_w2v)}")

print("\n  Méthode 4 : Modèle LDA")
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X_tfidf) 

for topic_idx, topic in enumerate(lda.components_):
    top_words = [mots_tfidf[i] for i in topic.argsort()[:-5:-1]]
    print(f"Sujet LDA {topic_idx + 1} : {', '.join(top_words)}")

print("\nFin")
