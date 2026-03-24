import json
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 1. CARICA TEST LOO
# =========================

with open("/home/marino/tesi/dataset_training/cross_validation_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Numero esempi test:", len(test_data))

# =========================
# 2. ESTRAI PROFILI E TARGET
# =========================

all_profiles = [item["profilo_utente"] for item in test_data]
all_targets = [item["target"] for item in test_data]

# ordine deterministico
unique_targets = sorted(list(set(all_targets)))

print("Numero target unici:", len(unique_targets))

# =========================
# 3. TF-IDF VECTORIZER
# =========================

print("Fit TF-IDF sui target...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,   # puoi mettere "english" se vuoi
    max_features=50000
)

target_tfidf = vectorizer.fit_transform(unique_targets)

print("Trasformo profili...")
profile_tfidf = vectorizer.transform(all_profiles)

# =========================
# 4. MATRICE DI SIMILARITÀ
# =========================

print("Calcolo similarità coseno...")

similarity_matrix = cosine_similarity(profile_tfidf, target_tfidf)

# Convertiamo in tensor per riusare il tuo codice
similarity_matrix = torch.tensor(similarity_matrix)

# =========================
# 5. SALVATAGGIO (facoltativo)
# =========================

torch.save({
    "similarity_matrix": similarity_matrix,
    "unique_targets": unique_targets
}, "/home/marino/tesi/similarity_data_tfidf.pt")

print("Matrice TF-IDF salvata.")

# =========================
# 6. FUNZIONE HIT RATE
# =========================

def compute_hit_rate(similarity_matrix, test_data, unique_targets, K):

    hits = 0
    target_to_index = {t: i for i, t in enumerate(unique_targets)}

    for i, item in enumerate(test_data):

        correct_target = item["target"]
        correct_index = target_to_index[correct_target]

        top_k_indices = torch.topk(similarity_matrix[i], K).indices

        if correct_index in top_k_indices:
            hits += 1

    return hits / len(test_data)

# =========================
# 7. CALCOLO HR
# =========================

hr10 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 10)
hr20 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 20)
hr50 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 50)

print(f"HR@10: {hr10:.4f}")
print(f"HR@20: {hr20:.4f}")
print(f"HR@50: {hr50:.4f}")