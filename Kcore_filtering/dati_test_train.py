import json
import pandas as pd

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

# ==============================
# FUNZIONI UTILI
# ==============================
def intersection_size(set_a, set_b):
    return len(set_a & set_b)

def review_set(df):
    return set(
        zip(df["user_id"], df["product_id"], df["date"])
    )

# ==============================
# SET DI GIOCHI
# ==============================
items_train1 = set(df_train1_after["product_id"])
items_train2 = set(df_train2_after["product_id"])
items_test   = set(df_test_after["product_id"])

# ==============================
# SET DI UTENTI
# ==============================
users_train1 = set(df_train1_after["user_id"])
users_train2 = set(df_train2_after["user_id"])
users_test   = set(df_test_after["user_id"])

# ==============================
# SET DI RECENSIONI
# ==============================
reviews_train1 = review_set(df_train1_after)
reviews_train2 = review_set(df_train2_after)
reviews_test   = review_set(df_test_after)

# ==============================
# STAMPA RISULTATI
# ==============================
print("\n🎮 OVERLAP GIOCHI")
print("Train1 ∩ Train2:", intersection_size(items_train1, items_train2))
print("Train1 ∩ Test  :", intersection_size(items_train1, items_test))
print("Train2 ∩ Test  :", intersection_size(items_train2, items_test))

print("\n👤 OVERLAP UTENTI")
print("Train1 ∩ Train2:", intersection_size(users_train1, users_train2))
print("Train1 ∩ Test  :", intersection_size(users_train1, users_test))
print("Train2 ∩ Test  :", intersection_size(users_train2, users_test))

print("\n📝 OVERLAP RECENSIONI")
print("Train1 ∩ Train2:", intersection_size(reviews_train1, reviews_train2))
print("Train1 ∩ Test  :", intersection_size(reviews_train1, reviews_test))
print("Train2 ∩ Test  :", intersection_size(reviews_train2, reviews_test))
