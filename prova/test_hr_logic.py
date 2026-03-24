import torch

def compute_vectorized_hit_rate(similarity_matrix, labels, unique_targets, Ks=[10, 20, 50]):
    target_to_index = {t: i for i, t in enumerate(unique_targets)}
    y_true = torch.tensor([target_to_index[t] for t in labels], device=similarity_matrix.device)
    max_k = max(Ks)
    _, top_k_indices = torch.topk(similarity_matrix, max_k, dim=1)
    results = {}
    for k in Ks:
        hits = (top_k_indices[:, :k] == y_true.view(-1, 1)).any(dim=1)
        hr = hits.float().mean().item()
        results[k] = hr
    return results

def test_hr():
    # 3 queries, 5 possible targets
    unique_targets = ["A", "B", "C", "D", "E"]
    # Labels for queries: 0 -> A, 1 -> B, 2 -> E
    labels = ["A", "B", "E"]
    
    # Sim matrix (3, 5)
    sim = torch.tensor([
        [0.9, 0.1, 0.2, 0.3, 0.4], # Top1: A (Hit), Top2: A, E
        [0.1, 0.8, 0.2, 0.3, 0.4], # Top1: B (Hit), Top2: B, E
        [0.1, 0.2, 0.3, 0.4, 0.5]  # Top1: E (Hit), Top2: E, D
    ])
    
    res = compute_vectorized_hit_rate(sim, labels, unique_targets, Ks=[1, 2])
    print(f"Test 1 Results: {res}")
    assert res[1] == 1.0
    assert res[2] == 1.0

    # Case with misses
    labels2 = ["C", "D", "A"]
    # Query 0: C is at index 2 (val 0.2). Top1 is A (0.9). Top2 is A, E. Not in Top2.
    # Query 1: D is at index 3 (val 0.3). Top1 is B (0.8). Top2 is B, E. Not in Top2.
    # Query 2: A is at index 0 (val 0.1). Top1 is E (0.5). Top2 is E, D. Not in Top2.
    res2 = compute_vectorized_hit_rate(sim, labels2, unique_targets, Ks=[1, 2])
    print(f"Test 2 Results: {res2}")
    assert res2[1] == 0.0
    assert res2[2] == 0.0

    print("All tests passed!")

if __name__ == "__main__":
    test_hr()
