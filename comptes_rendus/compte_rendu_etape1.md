# NLP Mental Health Conversations
## Compte-rendu technique — Étape 1 : Exploration du jeu de données

> **Projet** : Analyse de conversations patient-thérapeute sur la santé mentale  
> **Encadrante** : Lynda Tamine-Lechani  
> **Équipe** : Ulysse Chasseigne, Idir Yahiaoui, Bantsoukissa Laure, Cruz Fernandes Diogo, Emin Belkheir, Esso-Y-N Trésor PANA, Logan Larroux, Yvan Sellier, Victor Blanquart–Blanchet, Lucas Da Silva

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Analyse statistique](#2-analyse-statistique)
3. [Analyse lexicale — Mots les plus fréquents](#3-analyse-lexicale--mots-les-plus-fréquents)
4. [Topic Modeling — Méthode 1 : Lexique personnalisé](#4-topic-modeling--méthode-1--lexique-personnalisé)
5. [Topic Modeling — Méthode 2 : TF-IDF + K-Means](#5-topic-modeling--méthode-2--tf-idf--k-means)
6. [Topic Modeling — Méthode 3 : Word Embeddings](#6-topic-modeling--méthode-3--word-embeddings)
7. [Topic Modeling — Méthode 4 : LDA](#7-topic-modeling--méthode-4--lda)
8. [Conclusion comparée des méthodes](#8-conclusion-comparée-des-méthodes)
9. [Problèmes rencontrés et questions ouvertes](#9-problèmes-rencontrés-et-questions-ouvertes)

---

## 1. Introduction

L'étape 1 a pour objectif d'explorer et de comprendre la structure du corpus de conversations patient-thérapeute issu du dataset Kaggle [`train.csv`]. Le fichier contient deux colonnes : le message du patient (**Context**) et la réponse du thérapeute (**Response**).

Chaque membre de l'équipe a travaillé de façon **individuelle** sur ses propres scripts Python afin de pouvoir comparer les résultats et les approches ensuite en groupe. Cette démarche volontairement divergente permet d'identifier les biais liés aux choix de prétraitement et de paramétrage.

> **Remarque importante** : Les prétraitements appliqués au texte ont un impact **direct et significatif** sur l'ensemble des résultats obtenus. La qualité du nettoyage (suppression des stopwords, normalisation, détection de la langue, filtrage des doublons) est donc un point critique qui explique la majorité des écarts observés entre les membres de l'équipe.

---

## 2. Analyse statistique

### 2.1 Prétraitement des données

Chaque membre a adopté une stratégie de nettoyage plus ou moins stricte. Voici les deux exemples d'approches :

**Approche minimaliste (Emin)** — suppression des lignes vides uniquement :

```python
# Suppression des lignes vides
df = df.dropna(how='all')
df = df.dropna(subset=['Context'])
df = df.dropna(subset=['Response'])
df = df.reset_index(drop=True)
```

**Approche étendue (Diogo)** — nettoyage complet avec détection de langue :

```python
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    words = text.strip().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def is_english(text):
    try:
        return detect(str(text)) == 'en'
    except:
        return False

# Pipeline de nettoyage complet
df = df.dropna(subset=["Response"])
df = df.drop_duplicates(subset=["Context", "Response"], keep="first")
df['Context'] = df['Context'].apply(clean_text)
df['Response'] = df['Response'].apply(clean_text)
mask = df['Context'].apply(is_english) & df['Response'].apply(is_english)
df = df[mask].reset_index(drop=True)
```

---

### 2.2 Résultats obtenus

Les divergences de prétraitement expliquent les écarts dans les résultats ci-dessous.

#### Nombre de patients uniques

| Résultat | Membres | Méthode appliquée |
|---|---|---|
| **995** | 8 personnes | Sans prétraitement — `nunique()` brut |
| **819** | 1 personne | Prétraitement + filtre sur les mots espagnols |
| **796** | 1 personne | Prétraitement + filtre sur les mots espagnols |

#### Nombre de thérapeutes uniques

| Résultat | Membres | Méthode appliquée |
|---|---|---|
| **2 479** | 6 personnes | Sans prétraitement |
| **2 472** | 1 personne | Léger prétraitement (suppression des retours à la ligne) |
| **2 409** | 1 personne | Prétraitement complet |
| **≥ 12** | 1 personne | Pattern matching (identification par motifs) |

> ⚠️ **Limite importante** : Les valeurs autour de 2 400-2 479 correspondent très probablement au **nombre de lignes/réponses** dans le fichier, et non au nombre réel de thérapeutes distincts. En l'absence d'identifiant unique (ID), il est impossible de différencier les thérapeutes entre eux avec certitude.

**Question ouverte :** Existe-t-il une technique fiable permettant de différencier patients et thérapeutes sans passer par un identifiant explicite ?

---

### 2.3 Nombre de mots moyen par patient / thérapeute

Ces chiffres sont **fortement impactés** par le niveau de prétraitement appliqué. Les membres ayant supprimé davantage de mots (stopwords, contractions, etc.) obtiennent naturellement des moyennes plus basses.

#### Côté patients

| Membre | Mots moyens |
|---|---|
| Victor | 66,86 |
| Emin | 55,2 |
| Idir | 24,8 |
| Lucas | 21,3 |
| Logan | 16,3 |

#### Côté thérapeutes

| Membre | Mots moyens |
|---|---|
| Emin | 177,2 |
| Victor | 140,2 |
| Lucas | 82,4 |
| Idir | 84,8 |
| Logan | 60,1 |

> On observe un **rapport allant du simple au quadruple** selon le niveau de filtrage. Les thérapeutes produisent systématiquement des réponses bien plus longues que les messages des patients, ce qui est cohérent avec la nature de l'échange thérapeutique.

---

## 3. Analyse lexicale — Mots les plus fréquents

### 3.1 Sans filtrage renforcé

| Côté patients | Côté thérapeutes | Global |
|---|---|---|
| *im, feel, dont* | *feel, may, help, like* | *feel, like, may* |

Ces résultats, bien que représentatifs du corpus brut, sont peu exploitables : les mots les plus fréquents sont des **mots fonctionnels** ou des contractions sans valeur sémantique forte.

### 3.2 Avec filtrage renforcé (stopwords personnalisés)

Certains membres ont étendu la liste des stopwords pour exclure des mots peu informatifs mais non présents dans les listes standards :

```python
# Exemple — Logan
extra_stopwords = {
    "feel", "like", "want", "know", "time",
    "thing", "good", "need", "year", "think",
    "tell", "say", "come", "go"
}
```

```python
# Exemple — Lucas
mots_inutiles = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    # ... [liste complète dans le notebook]
    "feel", "really", "things", "just", "like", "very", "too", "lot",
    "there", "back", "even", "every", "years", "told", "going", "still"
])
```

| Côté patients | Côté thérapeutes | Global |
|---|---|---|
| *relationship, love, anxiety* | *relationship, like* | *relationship, help* |

### 3.3 Interprétation

Les deux niveaux de filtrage sont **complémentaires** et révèlent des niveaux d'information différents :

- **Sans filtrage renforcé** → on observe les **expressions émotionnelles** brutes des patients (`feel`, `love`) et la posture d'aide des thérapeutes (`may`, `help`, `would`)
- **Avec filtrage renforcé** → on accède aux **thématiques explicites** : `relationship`, `anxiety`, `love` côté patient ; `relationship` côté thérapeute

> Ces résultats confirment que le corpus est centré sur les **relations interpersonnelles**, sujet le plus fréquent, couplé à des états psychologiques comme l'anxiété.

**Question ouverte :** Existe-t-il une méthode automatique pour pondérer les mots selon leur pertinence sémantique, sans les supprimer manuellement ?

L'analyse lexicale donne ainsi une première image du corpus, ce dont les patients parlent et comment les thérapeutes leur répondent. Pour aller plus loin et identifier des **thématiques structurées** au sein de ces échanges, **quatre méthodes** de topic modeling ont été appliquées et comparées.

---

## 4. Topic Modeling — Méthode 1 : Lexique personnalisé

### 4.1 Principe

Un dictionnaire thématique est construit **manuellement**, en associant chaque thème de santé mentale à une liste de mots-clés. Chaque question peut être assignée à **plusieurs thèmes simultanément** (classification multi-label).

```python
# Exemple — Trésor
lexique = {
    'Anxiété':    ['anxious', 'anxiety', 'panic', 'fear'],
    'Dépression': ['depress', 'depression', 'sad', 'suicide'],
    'Relation':   ['relationship', 'husband', 'wife', 'partner']
}

# Exemple — Diogo (version étendue)
LEXICON = {
    'Anxiety':    ['anxiety', 'anxious', 'panic', 'worry', 'worried', 'nervous',
                   'fear', 'phobia', 'stress', 'stressed', 'overthink', 'dread'],
    'Depression': ['depression', 'depressed', 'sad', 'sadness', 'hopeless',
                   'empty', 'numb', 'unmotivated', 'despair', 'grief', 'miserable'],
    'Relationships': ['relationship', 'partner', 'boyfriend', 'girlfriend', 'husband',
                      'wife', 'marriage', 'divorce', 'breakup', 'couple', 'dating',
                      'love', 'trust'],
    'Family':     ['family', 'mother', 'father', 'parent', 'child', 'children',
                   'son', 'daughter', 'sister', 'brother'],
    'Self-esteem':['self-esteem', 'confidence', 'worthless', 'shame', 'guilt',
                   'insecure', 'identity', 'self-worth', 'self-image', 'inadequate'],
    'Suicidal':   ['suicide', 'suicidal', 'kill myself', 'end my life', 'self-harm',
                   'cutting', 'hurt myself', 'die', 'death'],
    'Addiction':  ['addiction', 'addicted', 'alcohol', 'drugs', 'substance',
                   'drinking', 'smoking', 'marijuana', 'weed', 'cocaine', 'gambling'],
    'Trauma/PTSD':['trauma', 'ptsd', 'abuse', 'assault', 'flashback', 'nightmare',
                   'violence', 'rape', 'harassment', 'survivor'],
    'Sleep':      ['sleep', 'insomnia', 'tired', 'exhausted', 'fatigue', 'rest',
                   'awake', 'sleepless'],
    'Anger':      ['anger', 'angry', 'rage', 'furious', 'irritable', 'frustration',
                   'aggressive', 'temper'],
    'Loneliness': ['lonely', 'loneliness', 'alone', 'isolated', 'isolation',
                   'disconnected', 'friendless'],
    'Work/School':['work', 'job', 'career', 'school', 'college', 'university',
                   'boss', 'coworker', 'burnout', 'performance', 'exam', 'grades']
}
```

### 4.2 Résultats

Les thèmes apparaissant de façon **transversale** dans toutes les conversations sont :

- **Anxiété** (`anxiety`, `panic`, `stress`)
- **Dépression** (`depression`, `sad`, `hopeless`)
- **Relations** (`relationship`, `love`, `partner`) — thème le plus fréquent

> Avoir le thème *Relation* comme le plus fréquent peut sembler logique au vue du contexte du Dataframe, mais sela ne veut pas dire qu'il n'existe pas d'autres thèmes bien plus important et auquels on ne penserait pas directement.

### 4.3 Évaluation critique

| Avantage | Limite |
|---|---|
| Très **interprétable** et explicable | **Rigide** : ne capte que les mots explicitement listés |
| Facile à adapter au domaine | Ne gère pas les **synonymes** non prévus |
| Résultats stables et reproductibles | Nécessite une **expertise métier** pour construire le lexique |
| Classification multi-label naturelle | Insensible au **contexte** d'usage d'un mot |

---

## 5. Topic Modeling — Méthode 2 : TF-IDF + K-Means

### 5.1 Principe

Chaque texte est transformé en un **vecteur numérique** via TF-IDF (Term Frequency–Inverse Document Frequency), où chaque dimension représente l'importance relative d'un mot dans un document par rapport à l'ensemble du corpus. Ces vecteurs sont ensuite regroupés par l'algorithme **K-Means**.

### 5.2 Sélection du nombre de clusters (k)

Le choix de `k` est délicat. Plusieurs stratégies ont été explorées :

- **Définition manuelle** : observation visuelle des clusters pour juger de leur pertinence
- **Méthode du coude** (*elbow method*) : on trace l'inertie en fonction de `k` et on cherche le point de rupture
- **Équilibrage des tailles** : minimiser les écarts de taille entre clusters

La méthode du coude suggère `k = 6`, mais la courbe décroît faiblement à partir de ce point, ce qui rend le choix ambigu. Visuellement, `k = 4` ou `k = 8` seraient aussi défendables.

### 5.3 Résultats comparés selon k

#### K = 6 — Clusters déséquilibrés — Logan

| Thème identifié | Effectif |
|---|---|
| Dépression sévère — idées suicidaires | 457 |
| Mal-être général — perte de motivation | 112 |
| Dépression sévère — fatigue mentale | 46 |
| Thérapie de couple — anxiété | 223 |
| Traumatisme — abus — maladie grave | 1 718 |
| Problèmes familiaux — divorce — identité | 139 |

> ⚠️ Résultat insuffisant : fort déséquilibre de taille (46 vs 1718), thèmes qui se recoupent.

#### K = 5 — Clusters partiellement pertinents — Emin

```
years - told - sex - time - back - child
im - dont - feel - know - like - want
therapy - normal - still - shaky - everytime - cry
decide - client - counselor - end - terminate - working
counseling - address - history - many - issues - process
```

> Un cluster contient des mots peu informatifs, et les deux derniers sont trop similaires.

#### K = 4 — Clusters trop vagues — Victor

```
know, want, dont, years, get, relationship, sex, told, time, love
therapy, sessions, normal, still, counseling, cry, decide, client, counselor, process
address, history, many, issues, counseling, ive, breast, insomniac, lifetime, happily
feel, like, dont, people, really, know, always, get, anything, cant
```

> Trop peu précis : le 4e cluster ne contient rien d'utile, les 2e et 3e ne se distinguent pas assez.

#### K = 3 — Meilleure configuration avec LSA

En appliquant une **décomposition LSA** (Latent Semantic Analysis via SVD) sur la matrice TF-IDF avant le clustering, et avec des hyperparamètres plus fins :

```python
vec = TfidfVectorizer(
    max_features=10000,                     # limite aux 10000 termes les plus fréquents  
    ngram_range=(1, 3),                     # unigrammes, bigrammes et trigrammes
    min_df=3,                               # ignore les mots trop rares
    max_df=0.5,                             # ignore les mots trop fréquents
    sublinear_tf=True,                      # normalisation logarithmique
    stop_words=list(STOPWORDS),             # retire une liste de mots supplémentaire
    strip_accents="unicode",                # normalise les accents (é -> e, à -> a etc.)
    token_pattern=r"(?u)\b[a-zA-Z]{2,}\b"   # ne conserve que les séquence alphabétique d'au moins
                                            #   2 caractères
)
```

| Topic | Effectif | Part | Mots-clés principaux |
|---|---|---|---|
| Family and relational dynamics | 309 | 38,8 % | friends, family, child, dad, mom, husband |
| Emotional distress and mental health issues | 284 | 35,7 % | anxiety, depression, stop, start, constantly, school |
| Romantic relationships and intimacy | 203 | 25,5 % | love, relationship, boyfriend, sex, hes, girl |

> ✅ Meilleur résultat TF-IDF : thèmes distincts, tailles équilibrées, couvrant 100 % du corpus.

### 5.4 Évaluation critique

| Avantage | Limite |
|---|---|
| Méthode **rapide** et scalable | Sensible au choix de `k` |
| Bien documentée et reproductible | Ignore la **sémantique contextuelle** des mots |
| Compatible avec LSA pour de meilleurs résultats | Les mots sont traités comme des entités **indépendantes** |

**Question ouverte :** Existe-t-il une méthode plus robuste que le coude pour déterminer automatiquement le nombre optimal de clusters ?

---

## 6. Topic Modeling — Méthode 3 : Word Embeddings
Le Word Embedding (plongement lexical) est une **représentation sémantique des mots** sous forme de **vecteurs** de nombres réels. Deux mots de sens proches ont des vecteurs proches dans l'espace méthématique.  

Trois approches de word embeddings ont été testées et comparées.

### 6.1 Word2Vec (Skip-gram) — Logan
Contrairement à TF_IDF qui est une représentation statistique (fréquence d'apparition), Word2Vec est une représentation sémantique, il apprend des représentations vectorielles à partir du **contexte local** de chaque mot (par exemple *sad* et *depressed* se trouvent proches dans l'espace vectoriel parce qu'il apparaissent dans des contextes similaires).  
Le mode **Skip-gram** (prédit le contexte à partir d'un mot) est préféré au **CBOW** (prédit un mot depuis son contexte) car notre corpus est **petit et spécialisé** : Skip-gram est plus performant dans ce cas.

```python
model = Word2Vec(
    sentences=all_sentences,
    vector_size=100,   # dimensions des vecteurs
    window=5,          # contexte de 5 mots de chaque côté
    min_count=3,       # ignore les mots apparaissant < 3 fois
    workers=4,         # calcul parallèle sur 4 threads
    epochs=10,         # indique 10 passages complet sur le corpus
    seed=42,           # permet de reproduire le résultat
    sg=1               # Skip-gram (0 = CBOW)
)
```
> Il est important de préciser que le paramètre ```epoch``` ne doit pas être trop grand, cela augmanterais le risque de **surapprentissage** sur un corpus petit comme le notre.


**Résultats avec K = 6 (patients) :**

| Cluster | Effectif | Thème |
|---|---|---|
| 0 | 893 | Dépression |
| 1 | 21 | Problèmes sexuels — santé physique |
| 2 | 414 | Thérapie de couple — anxiété |
| 3 | 1 084 | Mal-être général |
| 4 | 195 | Anxiété — trouble du sommeil |
| 5 | 88 | Traumatisme |

> La répartition est déséquilibrée, mais cela peut être interprété positivement : le cluster 1 isole des problèmes sexuels spécifiques qui auraient été noyés dans un cluster "relations de couple" plus large.

---

### 6.2 Sentence Transformer (BERT / all-MiniLM-L6-v2) — Idir

Contrairement à Word2Vec qui produit un vecteur **par mot**, Sentence Transformer produit un vecteur **par phrase entière** en 384 dimensions, via un modèle pré-entraîné sur de larges corpus de langue naturelle.

**Résultats avec K = 8 :**

| Effectif | Thème du cluster | Mots représentatifs |
|---|---|---|
| 111 | Marriage and infidelity | husband, married, relationship, hes, love, wife |
| 86 | Seeking therapy and abuse | therapist, counseling, child, therapy, counselor, issues |
| 121 | Romantic relationships and dating | love, relationship, boyfriend, guy, dating, hes |
| 101 | Family and parenting | mom, dad, child, mother, shes, family |
| 95 | Anxiety and mental disorder | anxiety, depression, attacks, sleep, panic, disorder |
| 105 | Social anxiety and communication | thoughts, friends, talking, upset, anger, fear |
| 75 | Sexuality and gender identity | sex, girl, love, men, afraid, girls |
| 102 | Depression and social life | friends, school, sad, depressed, family, talk |

> ✅ **Meilleur résultat** de la méthode 3 : clusters de taille homogène (~75–121), thèmes distincts et bien interprétables. Certains mots apparaissent dans plusieurs clusters (`love`, `relationship`) mais le sens varie selon les co-occurrents.

---

### 6.3 spaCy (embeddings moyennés) — Emin

spaCy calcule un vecteur de phrase en **moyennant les vecteurs des mots** (contrairement aux transformers qui prennent en compte le contexte global).

**Résultats avec K = 5 :**

| Cluster | Mots principaux |
|---|---|
| 0 | counseling, therapist, know, end, client, counselor |
| 1 | dont, im, like, feel, know, want |
| 2 | counseling, really, anything, help, people, something |
| 3 | im, ive, like, feel, get, many |
| 4 | feel, years, time, im, like, relationship |

> ⚠️ Résultats insuffisants : clusters 0 et 2 très similaires, plusieurs clusters dominés par des mots peu informatifs.

---

### 6.4 Bilan comparatif — Méthode 3

| Modèle | Qualité des clusters | Points forts | Limites |
|---|---|---|---|
| **Sentence Transformer** | ✅ Très bonne | Contexte global, tailles homogènes | Modèle lourd, moins interprétable |
| **Word2Vec (Skip-gram)** | ✅ Bonne | Granularité thématique fine | Déséquilibre de taille |
| **spaCy** | ⚠️ Passable | Rapide, simple | Moyenne de vecteurs = perte de contexte |

> **Conclusion** : Sentence Transformer est la méthode la plus robuste pour ce corpus.

---

## 7. Topic Modeling — Méthode 4 : LDA

### 7.1 Principe

LDA (*Latent Dirichlet Allocation*) est un modèle probabiliste génératif. Contrairement au clustering, LDA produit une **assignation douce** : chaque document reçoit une **distribution de probabilités** sur l'ensemble des topics (et non une assignation exclusive à un seul cluster).

- Chaque **document** est un mélange de plusieurs sujets
- Chaque **sujet** est défini par une distribution de mots
- LDA identifie des groupes de mots qui **co-apparaissent** fréquemment
- Elle associe ensuite à chaque document une **probabilité d'appartenance** à chaque topic

```python
model = LdaModel(
    corpus=corpus,          # corpus au format Bag-of-Words
    id2word=dictionary,     # mapping id -> mot
    num_topics=n_topics,    # nombre de topics (à ajuster)
    random_state=42,        # reproductibilité
    passes=passes           # passes sur le corpus pour la convergence
)
```

### 7.2 Résultats comparés

#### Résultats à 6 topics — Logan

| Thème | Mots représentatifs |
|---|---|
| Relations de couple et infidélité | relationship, husband, child, listen, cheat, get, woman, past |
| Relations familiales et soutien social | relationship, people, person, help, dad, make, family, past |
| Relations amoureuses et communication | love, friend, talk, boyfriend, right, therapist, work, try |
| Anxiété et gestion émotionnelle | anxiety, help, talk, friend, feeling, get, try, take |
| Détresse émotionnelle et conflits personnels | thought, boyfriend, girl, life, have, stop, cry, month |
| Thérapie et problématiques de couple | normal, issue, wife, therapy, walk, voice, couple, self |

#### Résultats à 6 topics — Diogo

| Thème | Mots représentatifs |
|---|---|
| Romantic and sexual issues | love, hes, sex, relationship, boyfriend |
| Mental health and disorders | anxiety, depression, shes, boyfriend, mom |
| Therapy, communication and coping | school, friends, right, high, live |
| Relationship trust and family | girlfriend, thinking, hes, thoughts, stop |
| Intrusive thoughts and past | mother, mom, dad, son, child |
| Family anger and conflict | therapist, money, phone, relationship, friends |

#### Résultats à 5 topics — Lucas

| Thème | Mots représentatifs |
|---|---|
| Santé mentale, sexualité et contexte familial | sexual, family, self, depression, married, anxiety, address, history |
| Relations amoureuses et questionnements affectifs | say, right, boyfriend, think, girlfriend, says, relationship, past, sex, love |
| Anxiété, vie quotidienne et thérapie | think, boyfriend, having, couple, school, anxiety, therapy, normal, feeling |
| Détresse émotionnelle et pensées négatives | bad, think, sex, relationship, afraid, help, thoughts, love, stop |
| Relations de couple, vie familiale et évolution personnelle | boyfriend, make, counseling, live, child, life, people, talk, relationship |

> **Note sur le chevauchement des topics** : des mots comme `relationship` ou `boyfriend` apparaissent dans plusieurs topics. Ce n'est pas nécessairement un défaut — LDA tolère cette ambiguïté par conception. Le sens du mot est précisé par les autres mots du topic. Par exemple, `relationship` peut désigner une relation familiale (avec `dad`, `child`) ou amoureuse (avec `boyfriend`, `love`).

### 7.3 Évaluation critique

| Avantage | Limite |
|---|---|
| Topics **interprétables** et cohérents | Choix du nombre de topics difficile |
| Assignation douce = **nuance** inter-topics | Sensible au prétraitement |
| Bien adapté aux textes courts et spécialisés | Résultats **variables** selon les hyperparamètres |
| Fondement probabiliste rigoureux | Ne capture pas le contexte long |

---

## 8. Conclusion comparée des méthodes

Aucune méthode n'est universellement supérieure. Elles sont **complémentaires** et répondent à des besoins différents.

| Méthode | Forces | Faiblesses | Usage recommandé |
|---|---|---|---|
| **Lexique manuel** | Interprétable, contrôlable | Rigide, non-généralisant | Catégorisation ciblée sur un domaine connu |
| **TF-IDF + K-Means** | Rapide, scalable | Ignore le contexte sémantique | Exploration initiale d'un corpus large |
| **Word2Vec** | Capture les similarités sémantiques | Déséquilibre de clusters possible | Représentation sémantique de mots |
| **Sentence Transformer** | Contexte global, clusters homogènes | Lourd, moins interprétable | Meilleure qualité de clustering |
| **LDA** | Assignation douce, très interprétable | Chevauchement de topics | Analyse thématique et interprétation |

### Recommandation globale

- Pour **interpréter** les données : **LDA** est le plus lisible et le plus informativement riche
- Pour **regrouper** les documents : **Sentence Transformer** (BERT) offre la meilleure qualité de clustering
- Pour une **catégorisation stricte** : le **lexique manuel** reste le plus fiable si le domaine est bien défini

---

## 9. Problèmes rencontrés et questions ouvertes

### 9.1 Problèmes identifiés

| Problème | Description | Impact |
|---|---|---|
| **Hétérogénéité du nettoyage** | Chaque membre a appliqué un pipeline différent | Résultats non directement comparables |
| **Gestion des doublons** | Nombre de patients variant de 796 à 995 selon les membres | Biais dans toutes les statistiques |
| **Commentaires insuffisants** | Manque de documentation dans les notebooks | Difficultés à comprendre le code des autres |
| **Choix du nombre de clusters** | Méthode du coude peu discriminante après K=6 | Difficulté à trouver une valeur de K optimale |
| **Normalisation des sorties** | Formats de résultats différents entre membres | Comparaison manuelle laborieuse |

### 9.2 Questions ouvertes à soumettre à l'encadrante

1. **Standardisation du nettoyage** : nos moyennes de mots varient du simple au quadruple selon le niveau de filtrage. Faut-il fixer une règle commune (mots bruts ou mots porteurs de sens uniquement) pour garantir la comparabilité des résultats ?

2. **Gestion des doublons** : le nombre de patients varie de 796 à 995. Y a-t-il une consigne stricte sur la suppression des messages courts ou des doublons ?

3. **Identification des thérapeutes** : existe-t-il une technique permettant de différencier les patients et thérapeutes entre eux sans identifiant ?

4. **Mots semi-informatifs** : existe-t-il une méthode automatique pour traiter des mots moins significatifs que d'autres mais pas considérés comme des stopwords standards ?

5. **Sélection du nombre de clusters** : existe-t-il une méthode plus robuste que la méthode du coude pour déterminer automatiquement `k` optimal ?

---