import json
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# ==============================
# Caricamento file
# ==============================
df_train2_base = load_jsonl("/home/marino/tesi/dataset_clean/train2.jsonl")
df_train2_enriched = load_jsonl("/home/marino/tesi/dataset_clean/train2_arrichito.jsonl")

# ==============================
# Statistiche TRAIN2 BASE
# ==============================
base_interactions = len(df_train2_base)
base_users = df_train2_base['user_id'].nunique()
base_items = df_train2_base['product_id'].nunique()

# ==============================
# Statistiche TRAIN2 ARRICCHITO
# ==============================
enr_interactions = len(df_train2_enriched)
enr_users = df_train2_enriched['user_id'].nunique()
enr_items = df_train2_enriched['product_id'].nunique()

# ==============================
# Stampa risultati
# ==============================
print("====== TRAIN 2 BASE ======")
print("Interazioni totali:", base_interactions)
print("Utenti unici:", base_users)
print("Giochi unici:", base_items)

print("\n====== TRAIN 2 ARRICCHITO ======")
print("Interazioni totali:", enr_interactions)
print("Utenti unici:", enr_users)
print("Giochi unici:", enr_items)


