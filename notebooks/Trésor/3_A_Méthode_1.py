lexique = {
    'Anxiété': ['anxious', 'anxiety', 'panic', 'fear'],
    'Dépression': ['depress', 'depression', 'sad', 'suicide'],
    'Relation': ['relationship', 'husband', 'wife', 'partner']
}

for sujet, mots in lexique.items():
    # On compte combien de contextes contiennent au moins un des mots du lexique
    presence = df_clean['Context'].str.lower().apply(lambda x: any(m in x for m in mots))
    print(f"Sujet {sujet} : présent dans {presence.sum()} conversations")