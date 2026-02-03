import json
import pandas as pd
from pathlib import Path

# === PATH BASE ===
BASE_DIR = Path(".")
CLEAN_DIR = BASE_DIR / "dataset_clean"

INTERACTIONS_CSV = CLEAN_DIR / "steam_games_interactions.csv"
GAMES_JSON = CLEAN_DIR / "steam_games_clean.json"

# === 1) CARICO FILE USER–GAME (user_id, game_id) ===
print(f"Carico interazioni da: {INTERACTIONS_CSV}")
df_inter = pd.read_csv(INTERACTIONS_CSV)

df_inter["user_id"] = df_inter["user_id"].astype(str)
df_inter["game_id"] = df_inter["game_id"].astype(str)

print(f"Interazioni caricate: {len(df_inter)}")

# === 2) CARICO FILE GIOCHI ===
print(f"\nCarico i metadati dei giochi da: {GAMES_JSON}")
with open(GAMES_JSON, "r", encoding="utf-8", errors="ignore") as f:
    games = json.load(f)

games_df = pd.DataFrame(games)

# Normalizzo id → game_id
games_df["game_id"] = games_df["id"].astype(str)

# Estraggo anno di rilascio del gioco
games_df["release_year"] = pd.to_datetime(
    games_df["release_date"], errors="coerce"
).dt.year # type: ignore

games_meta = games_df[["game_id", "release_year"]]

# === 3) MERGE TRA USER-GAME E RELEASE YEAR ===
merged = df_inter.merge(
    games_meta,
    on="game_id",
    how="left"     # manteniamo tutte le interazioni
)

# === 4) SALVATAGGIO ===
output_path = "/home/marino/tesi/dataset_clean/steam_games_interactions_with_year.csv"
merged.to_csv(output_path, index=False)

print(f"\nFile finale salvato in: {output_path}")
print("\nPrime righe del risultato:")
print(merged.head())
