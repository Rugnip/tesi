import pandas as pd
import json
from pathlib import Path

# === PATH ===
BASE_DIR = Path(".")
CLEAN_DIR = BASE_DIR / "dataset_clean"

INTERACTIONS = CLEAN_DIR / "steam_games_interactions.csv"
GAMES_JSON = CLEAN_DIR / "steam_games_clean.json"

# === 1) CARICO LE INTERAZIONI user_id–game_id ===
df_inter = pd.read_csv(INTERACTIONS)
df_inter["user_id"] = df_inter["user_id"].astype(str)
df_inter["game_id"] = df_inter["game_id"].astype(str)

# === 2) CARICO I GIOCHI CON LA DATA ===
with open(GAMES_JSON, "r", encoding="utf-8", errors="ignore") as f:
    games = json.load(f)

games_df = pd.DataFrame(games)
games_df["game_id"] = games_df["id"].astype(str)

# Estrai ANNO e MESE del rilascio del gioco
games_df["release_date"] = pd.to_datetime(games_df["release_date"], errors="coerce")
games_df["year"] = games_df["release_date"].dt.year
games_df["month"] = games_df["release_date"].dt.month

# Filtra solo i giochi rilasciati nel 2017–2018
games_df = games_df[games_df["year"].isin([2017, 2018])]

# Metadati utili (solo id + anno + mese)
games_meta = games_df[["game_id", "year", "month"]]

# === 3) JOIN: assegno a ogni interazione l’anno e il mese del gioco ===
merged = df_inter.merge(games_meta, on="game_id", how="inner")

# === 4) Calcolo UTENTI UNICI per (anno, mese) ===
result = (
    merged.groupby(["year", "month"])["user_id"]
          .nunique()
          .reset_index(name="unique_users")
          .sort_values(["year", "month"])
)

# 🔥 Convertiamo tutto a NUMERI INTERI
result["year"] = result["year"].astype(int)
result["month"] = result["month"].astype(int)
result["unique_users"] = result["unique_users"].astype(int)

# === 5) Salvataggio ===
OUTFILE = CLEAN_DIR / "unique_users_per_month_2017_2018.csv"
result.to_csv(OUTFILE, index=False)

print("Risultato:")
print(result)
print(f"\nFile salvato in: {OUTFILE.resolve()}")

