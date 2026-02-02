import pandas as pd
import json

# ==============================
# 1. Caricamento dati
# ==============================
input_path = "/home/marino/tesi/dataset_clean/steam_reviews_reduced.jsonl"
output_path = "/home/marino/tesi/dataset_clean/recensioni_kcore5.jsonl"

rows = []

with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # salta righe rotte

df = pd.DataFrame(rows)

print("File caricato correttamente")
print("Righe:", len(df))

print("=== PRIMA DEL K-CORE ===")
print("Interazioni totali:", len(df))
print("Utenti unici:", df['user_id'].nunique())
print("Giochi unici:", df['product_id'].nunique())


# ==============================
# 2. Funzione k-core
# ==============================
def k_core_filter(df, k=5):
    while True:
        user_counts = df['user_id'].value_counts()
        product_counts = df['product_id'].value_counts()

        bad_users = user_counts[user_counts < k].index
        bad_products = product_counts[product_counts < k].index

        if len(bad_users) == 0 and len(bad_products) == 0:
            break

        df = df[
            (~df['user_id'].isin(bad_users)) &
            (~df['product_id'].isin(bad_products))
        ]

    return df


df_kcore = k_core_filter(df, k=5)

print("\n=== DOPO IL K-CORE ===")
print("Interazioni totali:", len(df_kcore))
print("Utenti unici:", df_kcore['user_id'].nunique())
print("Giochi unici:", df_kcore['product_id'].nunique())


# ==============================
# 4. Controlli di correttezza
# ==============================
assert df_kcore['user_id'].value_counts().min() >= 5
assert df_kcore['product_id'].value_counts().min() >= 5

print("\n✔ Vincolo k-core rispettato")


# ==============================
# 5. Salvataggio file filtrato
# ==============================

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_kcore.iterrows():
        record = {
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "text": row["text"],
            "date": row["date"]
        }

        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("File salvato correttamente:", output_path)
