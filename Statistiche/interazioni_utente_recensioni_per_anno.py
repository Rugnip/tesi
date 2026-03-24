import pandas as pd

# =========================
# 1. Caricamento dati
# =========================

df = pd.read_csv("/home/marino/tesi/dataset_clean/steam_games_interactions_with_year.csv")

# =========================
# 2. Conteggio recensioni per (user, year)
# =========================

user_year_reviews = (
    df
    .groupby(["user_id", "release_year"])
    .size()
    .reset_index(name="num_reviews_year")
)

# =========================
# 3. Salvataggio file finale
# =========================

user_year_reviews.to_csv(
    "/home/marino/tesi/analysis_items/user_interaction_ratings_with_year.csv",
    index=False
)

print("File creato: user_reviews_per_year.csv")

# =========================
# 4. Controlli rapidi
# =========================

print(user_year_reviews.head())
print("\nUtenti unici:", user_year_reviews["user_id"].nunique())
print("Anni:", sorted(user_year_reviews["release_year"].unique()))
