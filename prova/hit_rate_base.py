import json
import torch
import time
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def load_data(json_path):
    """Carica il dataset di test."""
    print(f"Caricamento dati da {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Numero esempi test: {len(data)}")
    return data

def get_embeddings(model, texts, batch_size=32, desc="Encoding"):
    """Calcola embedding in modo efficiente."""
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True
    )

def compute_vectorized_hit_rate(similarity_matrix, labels, unique_targets, Ks=[10, 20, 50]):
    """
    Calcola l'Hit Rate in modo vettorizzato per diversi valori di K.
    
    Args:
        similarity_matrix (torch.Tensor): Matrice (N_queries, N_targets)
        labels (list): Lista dei target corretti per ogni query
        unique_targets (list): Lista ordinata dei target unici
        Ks (list): Valori di K per cui calcolare HR
    """
    target_to_index = {t: i for i, t in enumerate(unique_targets)}
    # Converti i target corretti in indici numerici
    y_true = torch.tensor([target_to_index[t] for t in labels], device=similarity_matrix.device)
    
    # Prendi il K massimo richiesto
    max_k = max(Ks)
    
    # Prendi i top-K indici per ogni query
    # top_k_indices: (N_queries, max_k)
    _, top_k_indices = torch.topk(similarity_matrix, max_k, dim=1)
    
    results = {}
    for k in Ks:
        # Controlla se l'indice corretto è tra i primi k
        hits = (top_k_indices[:, :k] == y_true.view(-1, 1)).any(dim=1)
        hr = hits.float().mean().item()
        results[k] = hr
        
    return results

def main():
    # =========================
    # CONFIGURAZIONE
    # =========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "unsloth/Qwen3-Embedding-0.6B"
    TEST_DATA_PATH = "/home/marino/tesi/dataset_training/cross_validation_test.json"
    SAVE_PATH = "/home/marino/tesi/similarity_data_base.pt"
    BATCH_SIZE = 8  # Ridotto per evitare OOM su GPU con poca memoria libera
    
    print(f"Utilizzo device: {DEVICE}")

    # =========================
    # 1. CARICA MODELLO E DATI
    # =========================
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    test_data = load_data(TEST_DATA_PATH)

    all_profiles = [item["profilo_utente"] for item in test_data]
    all_targets = [item["target"] for item in test_data]
    unique_targets = sorted(list(set(all_targets)))
    
    print(f"Numero target unici: {len(unique_targets)}")

    # =========================
    # 2. CALCOLA EMBEDDING
    # =========================
    print("\nCalcolo embedding profili...")
    profile_embeddings = get_embeddings(model, all_profiles, batch_size=BATCH_SIZE)

    # Libera memoria prima del prossimo encoding
    torch.cuda.empty_cache()

    print("Calcolo embedding target...")
    target_embeddings = get_embeddings(model, unique_targets, batch_size=BATCH_SIZE)

    # Libera memoria prima del calcolo della similarità
    torch.cuda.empty_cache()

    # =========================
    # 3. MATRICE DI SIMILARITÀ
    # =========================
    print("\nCalcolo matrice similarità...")
    start_time = time.time()
    # util.cos_sim gestisce già il calcolo su GPU se i tensori sono su GPU
    similarity_matrix = util.cos_sim(profile_embeddings, target_embeddings)
    
    # Portiamo su CPU per salvataggio e se la matrice è molto grande
    # ma eseguiamo il calcolo HR su GPU se possibile per velocità
    comp_time = time.time() - start_time
    print(f"Matrice calcolata in {comp_time:.2f} secondi.")

    # =========================
    # 4. CALCOLO HR (VETTORIZZATO)
    # =========================
    print("\nCalcolo Hit Rate...")
    hr_results = compute_vectorized_hit_rate(similarity_matrix, all_targets, unique_targets, Ks=[10, 20, 50])

    for k, val in hr_results.items():
        print(f"HR@{k}: {val:.4f}")

    # =========================
    # 5. SALVATAGGIO
    # =========================
    torch.save({
        "similarity_matrix": similarity_matrix.cpu(),
        "unique_targets": unique_targets,
        "hr_results": hr_results
    }, SAVE_PATH)
    
    print(f"\nRisultati salvati in: {SAVE_PATH}")

if __name__ == "__main__":
    main()
