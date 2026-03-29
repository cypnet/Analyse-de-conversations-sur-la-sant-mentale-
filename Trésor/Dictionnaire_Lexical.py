import re
from collections import Counter

# Liste simplifiée de mots vides (stop-words) anglais
STOP_WORDS = set(["i", "me", "my", "the", "and", "to", "a", "of", "it", "is", "in", "that", "you", "your", "was", "for"])

def top_mots(series, n=10):
    text = " ".join(series.astype(str)).lower()
    mots = re.findall(r'\b\w+\b', text)
    # Filtrage des mots vides et des mots trop courts
    mots_filtres = [m for m in mots if m not in STOP_WORDS and len(m) > 2]
    return Counter(mots_filtres).most_common(n)

print("Top mots Patients (Context) :", top_mots(df_clean['Context']))
print("Top mots Thérapeutes (Response) :", top_mots(df_clean['Response']))