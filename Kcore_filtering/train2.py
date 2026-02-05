import json
import pandas as pd

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

# ==============================
# 1. Carica train1 e test
# ==============================
df_train1 = load_jsonl("/home/marino/tesi/dataset_clean/users_80.jsonl")
df_test   = load_jsonl("/home/marino/tesi/dataset_clean/users_20.jsonl")

# ==============================
# 2. Item presenti nel test
# ==============================
products_test = set(df_test["product_id"].unique())

# ==============================
# 3. Costruzione train2
# ==============================
df_train2 = df_train1[
    ~df_train1["product_id"].isin(products_test)
]

# ==============================
# 4. Salvataggio JSONL
# ==============================
output_path = "/home/marino/tesi/dataset_clean/train2.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df_train2.iterrows():
        record = {
            "user_id": row["user_id"],
            "product_id": row["product_id"],
            "text": row["text"],
            "date": row["date"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
