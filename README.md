# 🧠 Analyse de conversations patient-thérapeute sur la santé mentale

> Projet universitaire — Université de Toulouse | IRIT | [Info5.BE-SHS]  
> Encadrante : **Lynda Tamine-Lechani**

---

## 📋 Description

Ce projet vise à analyser un corpus de conversations entre patients et thérapeutes afin de :
- Comprendre la structure et les thématiques des échanges
- Détecter les émotions et états psychologiques des patients (analyse de sentiments)
- Développer deux outils d'accès à l'information : un système de **résumé automatique** et un système de **question-réponse** sur la santé mentale

---

## 📁 Structure du dépôt

```
├── data/
│   └── train.csv               # Jeu de données Kaggle (conversations patient-thérapeute)
├── notebooks/                  # Notebooks de l'équipe
│   ├── CRUZDiogo/
│   ├── EMIN/
│   ├── LAURE/
│   ├── LOGAN/
│   ├── LUCAS/
│   ├── Trésor/
│   ├── Yvan/
│   └── idir/
├── comptes_rendus                  # Compte rendu pour chaque étapes   
│   ├── compte_rendu_etape1.md      # Compre rendu pour l'étape 1
│   ├── compte_rendu_etape2.md      # Compte rendu pour l'étape 2
│   └── compte_rendu_etape3/md      # Compte rendu pour l'étape 3
├── requirements.txt                # Bibliothèques python nécessaires 
└── README.md
```

---

## ⚙️ Installation

**Prérequis :** Python 3.9+

```bash
# Cloner le dépôt
git clone https://github.com/cypnet/Analyse-de-conversations-sur-la-sant-mentale.git
cd 

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Utilisation

Ce projet n'a pas pour but d'être exécuté dans sa totalité, celui-ci propose divers dossiers dans lesquels se situent les notebook Jupyter de l'équipe que vous pouvez exécuter à part.

Les notebooks Jupyter dans `notebooks/` contiennent les analyses détaillées et les visualisations associées à chaque étape.

---

## 🔬 Méthode

### Étape 1 — Exploration du jeu de données
- **Pré-traitement** : tokenisation, lemmatisation, suppression des stop-words
- **Statistiques descriptives** : nombre de patients, distribution des conversations, longueur moyenne des échanges
- **Analyse lexicale** : n-grammes, fréquences d'apparition par patient et par thérapeute
- **Topic Modeling** : comparaison de 4 approches (lexique manuel, TF-IDF + K-Means, Word2Vec/GloVe + clustering, LDA)

### Étape 2 — Analyse des sentiments 🚧 **En cours de construction** 🚧
- Quantification de la polarité et des émotions des échanges
- Comparaison des approches entre les deux groupes  



### Étape 3 — Outils d'accès à l'information 🚧 **En cours de construction** 🚧
| Groupe | Tâche |
|--------|-------|
| **Groupe 1** | Résumé automatique des conversations |
| **Groupe 2** | Système de question-réponse sur la santé mentale |  



---

## 📦 Données

Le jeu de données utilisé est publié sur [Kaggle](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations) et contient des conversations patient-thérapeute structurées en deux colonnes :
- `patient` : message du patient
- `therapist` : réponse du thérapeute

> ⚠️ Le fichier `train.csv` se situe dans chacuns de dossiers de chaque participants.

---

## 📅 Calendrier & Livrables

| Livrable | Date prévue | Statut |
|----------|-------------|--------|
| Cahier des charges | 15 mars 2026 | ✅ Rendu |
| Bilan de mi-parcours | 20 avril 2026 | 🔄 En cours |
| Livrable technique mensuel | Mensuel | 🔄 En cours |
| Rapport final + documentation technique | 1 juin 2026 | ⏳ À venir |

---

## 👥 Équipe

**Chef de projet :** Logan LARROUX

| Groupe 1 — Résumé des conversations | Groupe 2 — Q/R sur la santé mentale |
|--------------------------------------|--------------------------------------|
| Logan LARROUX (Chef de groupe) | Idir YAHIAOUI (Chef de groupe) |
| Ulysse CHASSEIGNE | Bantsoukissa LAURE |
| Cruz FERNANDES DIOGO | Esso-Y-N Trésor PANA |
| Emin BELKHEIR | Yvan SELLIER |
| Victor BLANQUART-BLANCHET | Lucas DA SILVA |

---

## 📄 Licence

Projet académique — Université de Toulouse, 2026. Usage éducatif uniqu