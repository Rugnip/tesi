import json
import re
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm

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
# 3. TOKENIZZAZIONE
# =========================

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

print("Tokenizzo target...")
tokenized_targets = [tokenize(t) for t in unique_targets]

# =========================
# 4. COSTRUISCI BM25
# =========================

bm25 = BM25Okapi(tokenized_targets)

# =========================
# 5. MATRICE DI SIMILARITÀ
# =========================

print("Calcolo matrice similarità BM25...")

similarity_matrix = []

for profile in tqdm(all_profiles):

    tokenized_profile = tokenize(profile)

    scores = bm25.get_scores(tokenized_profile)
    similarity_matrix.append(scores)

similarity_matrix = torch.tensor(similarity_matrix)

# =========================
# 6. SALVATAGGIO (facoltativo)
# =========================

torch.save({
    "similarity_matrix": similarity_matrix,
    "unique_targets": unique_targets
}, "/home/marino/tesi/similarity_data_bm25.pt")

print("Matrice BM25 salvata.")

# =========================
# 7. FUNZIONE HIT RATE
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
# 8. CALCOLO HR
# =========================

hr10 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 10)
hr20 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 20)
hr50 = compute_hit_rate(similarity_matrix, test_data, unique_targets, 50)

print(f"HR@10: {hr10:.4f}")
print(f"HR@20: {hr20:.4f}")
print(f"HR@50: {hr50:.4f}")