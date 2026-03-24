import torch

# =========================
# 1. CARICA DATI SALVATI
# =========================

data = torch.load("/home/marino/tesi/similarity_data_base.pt")

similarity_matrix = data["similarity_matrix"]
unique_targets = data["unique_targets"]

print("Shape similarity matrix:", similarity_matrix.shape)
print("Numero target nel catalogo:", len(unique_targets))


# =========================
# 2. FUNZIONE CATALOG COVERAGE
# =========================

def compute_catalog_coverage(similarity_matrix, num_targets, K):

    recommended_targets = set()

    for i in range(similarity_matrix.shape[0]):

        # top-K raccomandazioni per il profilo i
        top_k_indices = torch.topk(similarity_matrix[i], K).indices.tolist()

        # aggiungiamo al set globale
        recommended_targets.update(top_k_indices)

    coverage = len(recommended_targets) / num_targets

    return coverage


# =========================
# 3. CALCOLO COVERAGE
# =========================

cov10 = compute_catalog_coverage(similarity_matrix, len(unique_targets), 10)
cov20 = compute_catalog_coverage(similarity_matrix, len(unique_targets), 20)
cov50 = compute_catalog_coverage(similarity_matrix, len(unique_targets), 50)

print("\n===== CATALOG COVERAGE =====")
print(f"Coverage@10: {cov10:.4f}")
print(f"Coverage@20: {cov20:.4f}")
print(f"Coverage@50: {cov50:.4f}")