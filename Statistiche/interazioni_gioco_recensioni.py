import pandas as pd

# =========================
# 1. Caricamento dati
# =========================

# File delle interazioni (user_id, game_id)
interactions = pd.read_csv("/home/marino/tesi/dataset_clean/steam_games_interactions.csv")

# File dei giochi (metadata)
games = pd.read_json("/home/marino/tesi/dataset_clean/steam_games_clean.json")

# Controllo rapido
print("Interazioni:")
print(interactions.head())
print("\nGiochi:")
print(games.head())

# =========================
# 2. Conteggio interazioni per gioco
# =========================
# Numero di utenti distinti che hanno recensito ogni gioco

game_interactions_count = (
    interactions
    .groupby("game_id")["user_id"]
    .nunique()
    .reset_index()
    .rename(columns={"user_id": "num_interactions"})
)

print("\nConteggio interazioni per gioco:")
print(game_interactions_count.head())

# =========================
# 3. Merge con i metadata dei giochi
# =========================

final_df = game_interactions_count.merge(
    games,
    left_on="game_id",
    right_on="id",
    how="left"
)

# =========================
# 4. Riordino colonne (opzionale)
# =========================

cols_order = (
    ["game_id", "num_interactions"] +
    [c for c in final_df.columns if c not in ["game_id", "num_interactions", "id"]]
)

final_df = final_df[cols_order]

# =========================
# 5. Salvataggio file finale
# =========================

final_df.to_csv("/home/marino/tesi/analysis_items/game_interactions_with_ratings.csv", index=False)

print("\nFile salvato: game_interactions_with_rating.csv")

# =========================
# 6. Controlli utili
# =========================

print("\nTop 10 giochi più recensiti:")
print(final_df.sort_values("num_interactions", ascending=False).head(10))

print("\nNumero totale di giochi con almeno una recensione:")
print(final_df.shape[0])
