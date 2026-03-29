# NLP Mental Health Conversations  Topic Analysis

## Project Overview

This project focuses on analyzing a dataset of mental health conversations between users and psychologists using Natural Language Processing techniques. The dataset is anonymized , contains question-and-answer exchanges between users and experienced psychologists on various mental health topics. The objective of the project is to explore the data, build a lexical dictionary, and identify major topics as accurately as possible. The approach used in this project does not rely on a single method, but instead on multiple NLP techniques including rule-based lexicon matching, TF-IDF clustering, word embeddings, and probabilistic topic modeling.

----------

# Dataset

Dataset source: Kaggle — NLP Mental Health Conversations [https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations/data](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations/data)

The dataset contains:

-   Question-answer conversation pairs
-   Multiple users and psychologists
-   Context (user question) and Response (psychologist answer)

Two main datasets are used:

-   `df_users` → deduplicated dataset with one row per unique patient question
-   `df_therapists` → subset of responses containing identified therapist signatures

The goal is to identify major mental health topics from the patient questions.




# Methodology

The analysis method is based on multiple NLP approaches applied to patient questions. The main idea is that mental health conversations follow recurring thematic patterns. Therefore, topics can be identified using keyword matching, vector representations, and probabilistic modeling. The process consists of several steps:

### 1. Exploratory Data Analysis

The first step is to analyze the dataset:

-   Summary statistics (number of patients, therapists, exchanges)
-   Distribution of responses per patient
-   Exchange length statistics (words per question and response)
-   Therapist identification via signature extraction


This helps understand the structure and scale of the dataset before topic modeling.

----------

### 2. Lexical Dictionary

To understand vocabulary usage, a lexical dictionary is computed:

-   Word frequencies per conversation, per patient, per therapist
-   Top 50 most frequent words globally
-   Dominant words per group using patient/therapist frequency ratios


### 3. Topic Identification

Four methods are applied to identify major topics in patient questions:

**Method 1 — Custom Lexicon (rule-based)**

-   A domain-specific dictionary maps each mental health theme to a list of keywords
-   Each question is tagged with all matching topics (multi-label)
-   12 topics identified including Relationships, Anxiety, Depression, Family

**Method 2 — TF-IDF + KMeans clustering**

-   Patient questions are represented as TF-IDF vectors
-   LSA dimensionality reduction applied before clustering
-   KMeans groups similar questions into 12 clusters
-   Top keywords per cluster identify the topic

**Method 3 — SentenceTransformer + KMeans clustering**

-   Each question is encoded into a 384-dimensional semantic vector using `all-MiniLM-L6-v2`
-   KMeans clusters the embeddings into 8 groups
-   Captures semantic similarity — "anxious" and "worried" are treated as close

**Method 4 — LDA (Latent Dirichlet Allocation)**

-   Probabilistic generative model trained on word counts
-   Each question receives a probability distribution over topics (soft assignment)
-   3 macro-themes identified with high confidence (avg probability 0.87–0.89)
----------


### 4. Evaluation

The performance of each method is evaluated using:

-   Silhouette score (clustering methods)
-   Perplexity (LDA)
-   Human readability of top keywords per topic

The evaluation compares:

-   Topic coherence across methods
-   Multi-topic coverage (lexicon and LDA)
-   Keyword quality per cluster

This provides a measure of the relevance and distinctiveness of the identified topics.
    



----------


## Technologies Used

Python libraries used in this project:

-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn (TF-IDF, KMeans, LDA, SVD, silhouette score)
-   NLTK (stopwords)
-   LangDetect (language filtering)
-   SentenceTransformers (semantic embeddings)
-   Tabulate (formatted output)
    

----------
# Project Structure

```
project/
│
├── train.csv          # Raw dataset (download from Kaggle)
├── notebook.ipynb     # Main notebook — analysis and topic identification
└── README.md          # This file
```



----------

## Key Features of the Project

This project demonstrates:

-   Text preprocessing and cleaning
-   Lexical analysis and frequency statistics
-   Rule-based topic matching
-   Unsupervised clustering on text data
-   Semantic embeddings for NLP
-   Probabilistic topic modeling
-   Performance evaluation using silhouette score and perplexity

The approach shows how multiple NLP techniques can be combined and compared to solve a topic identification problem on mental health conversations.

----------


## Possible Improvements

Future improvements could include:

-   Using supervised classification with labeled topic data
-   Applying BERTopic for combined embedding and topic modeling
-   Using coherence scores (NPMI, CV) instead of perplexity for LDA evaluation
- Extending the analysis to therapist responses to identify response strategies
