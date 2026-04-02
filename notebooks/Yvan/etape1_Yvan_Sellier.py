import pandas as pd
import tkinter as tk
from tkinter import filedialog
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 1. SÉLECTION DU FICHIER
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()

print("Sélectionner le fichier train.csv...")
chemin_fichier = filedialog.askopenfilename(
    title="Sélectionne ton fichier train.csv",
    filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
)

# 2. CHARGEMENT ET NETTOYAGE
print("\n1. NETTOYAGE DES DONNÉES")
# supprime les cases vides et les lignes en double
df = pd.read_csv(chemin_fichier).dropna().drop_duplicates()

# 3. STATISTIQUES GLOBALES
nb_patients = df['Context'].nunique()
nb_therapists = df['Response'].nunique()
avg_exchanges = len(df) / nb_patients

print(f"Lignes conservées après nettoyage : {len(df)}")
print(f"Patients uniques                  : {nb_patients}")
print(f"Thérapeutes uniques               : {nb_therapists}")
print(f"Moyenne de réponses par patient   : {avg_exchanges:.2f}")

# 4. DICTIONNAIRE LEXICAL (FRÉQUENCE)
print("\n2. DICTIONNAIRE LEXICAL (TOP 10)")

def exporter_frequence_mots(text_series, filename):
    vec = CountVectorizer(stop_words='english', max_features=1000)
    matrix = vec.fit_transform(text_series)
    freqs = zip(vec.get_feature_names_out(), matrix.sum(axis=0).tolist()[0])
    df_freq = pd.DataFrame(freqs, columns=['Mot', 'Frequence']).sort_values(by='Frequence', ascending=False)
    df_freq.to_csv(filename, index=False)
    return df_freq.head(10)

# On ne compte le texte d'un patient qu'une seule fois
patients_uniques = df['Context'].drop_duplicates()

print("\nTop 10 Mots Patients :")
# cache la colonne des numéros inutiles
print(exporter_frequence_mots(patients_uniques, 'mots_patients_nettoyes.csv').to_string(index=False))

print("\nTop 10 Mots Thérapeutes :")
print(exporter_frequence_mots(df['Response'], 'mots_therapeutes_nettoyes.csv').to_string(index=False))

# 5. EXTRACTION DE SUJETS
print("\n3. IDENTIFICATION DES SUJETS (LDA)")

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(patients_uniques)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)

feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
    print(f"Sujet {topic_idx + 1} : {', '.join(top_words)}")
