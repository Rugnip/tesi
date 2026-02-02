import json
import random
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# ==============================
# 1. Caricamento dati
# ==============================
df_test = load_jsonl("/home/marino/tesi/dataset_clean/users_20.jsonl")
df_train1 = load_jsonl("/home/marino/tesi/dataset_clean/users_80.jsonl")
df_train2 = load_jsonl("/home/marino/tesi/dataset_clean/train2.jsonl")

print("Item test:", df_test['product_id'].nunique())
print("Item train1:", df_train1['product_id'].nunique())
print("Item train2 (base):", df_train2['product_id'].nunique())

# ==============================
# 2. Selezione 30% item dal test
# ==============================
items_test = df_test['product_id'].unique().tolist()
random.shuffle(items_test)

split_idx = int(0.3 * len(items_test))
items_to_copy = set(items_test[:split_idx])

print("Item copiati nel train2:", len(items_to_copy))

# ==============================
# 3. Copia interazioni (SOLO item selezionati)
# ==============================
df_from_test = df_test[df_test['product_id'].isin(items_to_copy)]
df_from_train1 = df_train1[df_train1['product_id'].isin(items_to_copy)]

print("Interazioni copiate dal test:", len(df_from_test))
print("Interazioni copiate dal train1:", len(df_from_train1))

# ==============================
# 4. Train2 arricchito (ADD)
# ==============================
df_train2_enriched = pd.concat(
    [df_train2, df_from_test, df_from_train1],
    ignore_index=True
)

# (opzionale ma consigliato)
df_train2_enriched = df_train2_enriched.drop_duplicates(
    subset=["user_id", "product_id", "date"]
)

# ==============================
# 5. Controlli
# ==============================
assert set(items_to_copy).issubset(
    set(df_train2_enriched['product_id'])
)

print("✔ Copia completata correttamente")

# ==============================
# 6. Salvataggio JSONL
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

save_jsonl(
    df_train2_enriched,
    "/home/marino/tesi/dataset_clean/train2_arrichito.jsonl"
)

print("File creato: train2_enriched.jsonl")
