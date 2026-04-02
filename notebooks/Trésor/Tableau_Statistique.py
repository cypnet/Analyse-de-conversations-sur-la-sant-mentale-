import pandas as pd

# Chargement
df = pd.read_csv('train.csv').dropna()

# 1. Identification des conversations uniques (par le texte du patient)
patients_uniques = df['Context'].nunique()

# 2. Nettoyage des doublons (Même question + Même réponse)
df_clean = df.drop_duplicates(subset=['Context', 'Response'])

# 3. Calcul de la moyenne d'échanges
# On divise le nombre total de lignes par le nombre de questions uniques
nb_total_echanges = len(df_clean)
moyenne_echanges = nb_total_echanges / patients_uniques

# 4. Estimation des thérapeutes par signature
def extraire_signature(text):
    derniere_ligne = str(text).split('\n')[-1].strip()
    # On cherche des titres professionnels courants en fin de message
    titres = ['Dr.', 'Therapist', 'Counselor', 'LPC', 'Ph.D', 'MD', 'LPCC']
    if len(derniere_ligne) < 60 and any(t in derniere_ligne for t in titres):
        return derniere_ligne
    return "Anonyme"

df_clean['Therapeute'] = df_clean['Response'].apply(extraire_signature)
nb_therapeutes = df_clean[df_clean['Therapeute'] != "Anonyme"]['Therapeute'].nunique()

print(f"Nombre de patients : {patients_uniques}")
print(f"Nombre de thérapeutes identifiés : {nb_therapeutes}")
print(f"Moyenne d'échanges par conversation : {moyenne_echanges:.2f}")