import json
import pandas as pd

# ==============================
# 1. Carica file 20% e 80%
# ==============================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

df_20 = load_jsonl("/home/marino/tesi/dataset_clean/users_20.jsonl")
df_80 = load_jsonl("/home/marino/tesi/dataset_clean/users_80.jsonl")

print("Utenti 20%:", df_20['user_id'].nunique())
print("Utenti 80%:", df_80['user_id'].nunique())


# ==============================
# 2. Trova i giochi del 20%
# ==============================
products_20 = set(df_20['product_id'].unique())

print("Giochi nel 20%:", len(products_20))


# ==============================
# 3. Rimuovi dal 80% i giochi presenti nel 20%
# ==============================
df_80_filtered = df_80[~df_80['product_id'].isin(products_20)]

print("\n=== DOPO RIMOZIONE ===")
print("Interazioni 80% originali:", len(df_80))
print("Interazioni 80% filtrate:", len(df_80_filtered))
print("Giochi rimasti:", df_80_filtered['product_id'].nunique())


# ==============================
# 4. Salvataggio JSONL safe
# ==============================
output_path = "/home/marino/tesi/dataset_clean/train2.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_80_filtered.iterrows():
        record = {
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "text": row["text"],
            "date": row["date"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\nFile creato:", output_path)
