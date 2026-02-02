import json
import random
import pandas as pd

# ==============================
# 1. Caricamento file k-core
# ==============================
input_path = "/home/marino/tesi/dataset_clean/recensioni_kcore5.jsonl"

rows = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

print("Interazioni totali:", len(df))
print("Utenti unici:", df['user_id'].nunique())


# ==============================
# 2. Split utenti 20% / 80%
# ==============================
users = df['user_id'].unique().tolist()
random.shuffle(users)

split_idx = int(0.2 * len(users))

users_20 = set(users[:split_idx])
users_80 = set(users[split_idx:])

df_20 = df[df['user_id'].isin(users_20)]
df_80 = df[df['user_id'].isin(users_80)]

print("\n=== SPLIT ===")
print("Utenti 20%:", df_20['user_id'].nunique())
print("Utenti 80%:", df_80['user_id'].nunique())


# ==============================
# 3. Salvataggio JSONL (safe)
# ==============================
def save_jsonl(df, path):
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "user_id": row["user_id"],
                "product_id": row["product_id"],
                "text": row["text"],
                "date": row["date"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


save_jsonl(df_20, "/home/marino/tesi/dataset_clean/users_20.jsonl")
save_jsonl(df_80, "/home/marino/tesi/dataset_clean/users_80.jsonl")

print("\nFile generati:")
print(" - users_20.jsonl")
print(" - users_80.jsonl")
