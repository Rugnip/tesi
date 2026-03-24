import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# =========================
# 1. CARICA MODELLO
# =========================

model = SentenceTransformer("/home/marino/tesi/models/qwen_lora")

# =========================
# 2. CARICA TEST LOO
# =========================

with open("/home/marino/tesi/dataset_training/cross_validation_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

print("Numero esempi test:", len(test_data))

# =========================
# 3. ESTRAI PROFILI E TARGET
# =========================

all_profiles = [item["profilo_utente"] for item in test_data]
all_targets = [item["target"] for item in test_data]

# ORDINE DETERMINISTICO
unique_targets = sorted(list(set(all_targets)))

print("Numero target unici:", len(unique_targets))

# =========================
# 4. CALCOLA EMBEDDING
# =========================

print("Calcolo embedding profili...")
profile_embeddings = model.encode(
    all_profiles,
    convert_to_tensor=True,
    show_progress_bar=True
)

print("Calcolo embedding target...")
target_embeddings = model.encode(
    unique_targets,
    convert_to_tensor=True,
    show_progress_bar=True
)

# =========================
# 5. MATRICE DI SIMILARITÀ
# =========================

print("Calcolo matrice similarità...")
similarity_matrix = util.cos_sim(profile_embeddings, target_embeddings)

# Portiamo su CPU per sicurezza
similarity_matrix = similarity_matrix.cpu()

# =========================
# 6. SALVATAGGIO CORRETTO
# =========================

torch.save({
    "similarity_matrix": similarity_matrix,
    "unique_targets": unique_targets
}, "/home/marino/tesi/similarity_data.pt")

print("Matrice salvata correttamente.")

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
