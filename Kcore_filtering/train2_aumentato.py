import json
import random
import pandas as pd

# ==============================
# Utils
# ==============================
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

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

print("Item selezionati:", len(items_to_copy))

# ==============================
# 3. Copia interazioni verso train2
# ==============================
df_from_test = df_test[df_test['product_id'].isin(items_to_copy)]
df_from_train1 = df_train1[df_train1['product_id'].isin(items_to_copy)]

print("Interazioni copiate dal test:", len(df_from_test))
print("Interazioni copiate dal train1:", len(df_from_train1))

df_train2_enriched = pd.concat(
    [df_train2, df_from_test, df_from_train1],
    ignore_index=True
).drop_duplicates(subset=["user_id", "product_id", "date"])

# ==============================
# 4. RIMOZIONE da test e train1
# ==============================
df_test_clean = df_test[~df_test['product_id'].isin(items_to_copy)]
df_train1_clean = df_train1[~df_train1['product_id'].isin(items_to_copy)]

print("Interazioni rimosse dal test:", len(df_test) - len(df_test_clean))
print("Interazioni rimosse dal train1:", len(df_train1) - len(df_train1_clean))

# ==============================
# 5. Controlli di sicurezza
# ==============================
assert set(items_to_copy).issubset(set(df_train2_enriched['product_id']))
assert set(items_to_copy).isdisjoint(set(df_test_clean['product_id']))
assert set(items_to_copy).isdisjoint(set(df_train1_clean['product_id']))

print("✔ Split consistente e pulito")

# ==============================
# 6. Salvataggio file finali
# ==============================
save_jsonl(
    df_train2_enriched,
    "/home/marino/tesi/dataset_clean/train2_arricchito.jsonl"
)

save_jsonl(
    df_train1_clean,
    "/home/marino/tesi/dataset_clean/users_80_pulito.jsonl"
)

save_jsonl(
    df_test_clean,
    "/home/marino/tesi/dataset_clean/users_20_pulito.jsonl"
)

print("✔ Tutti i file creati correttamente")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)

def get_stats(df):
    return {
        "interazioni": len(df),
        "utenti": df["user_id"].nunique(),
        "item": df["product_id"].nunique()
    }

def print_comparison(name, before, after):
    print(f"\n📊 {name}")
    print("-" * 50)
    print(f"{'':15} | {'PRIMA':>10} | {'DOPO':>10} | {'Δ':>10}")
    print("-" * 50)
    for k in before:
        delta = after[k] - before[k]
        print(f"{k:15} | {before[k]:10} | {after[k]:10} | {delta:+10}")

# ===== FILE ORIGINALI =====
df_test_before   = load_jsonl("/home/marino/tesi/dataset_clean/users_20.jsonl")
df_train1_before = load_jsonl("/home/marino/tesi/dataset_clean/users_80.jsonl")
df_train2_before = load_jsonl("/home/marino/tesi/dataset_clean/train2.jsonl")

# ===== FILE DOPO =====
df_test_after   = load_jsonl("/home/marino/tesi/dataset_clean/users_20_pulito.jsonl")
df_train1_after = load_jsonl("/home/marino/tesi/dataset_clean/users_80_pulito.jsonl")
df_train2_after = load_jsonl("/home/marino/tesi/dataset_clean/train2_arricchito.jsonl")

stats_test_before   = get_stats(df_test_before)
stats_test_after    = get_stats(df_test_after)

stats_train1_before = get_stats(df_train1_before)
stats_train1_after  = get_stats(df_train1_after)

stats_train2_before = get_stats(df_train2_before)
stats_train2_after  = get_stats(df_train2_after)

print_comparison("TEST (users_20)", stats_test_before, stats_test_after)
print_comparison("TRAIN 1 (users_80)", stats_train1_before, stats_train1_after)
print_comparison("TRAIN 2", stats_train2_before, stats_train2_after)

