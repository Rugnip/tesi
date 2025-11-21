import ast
from collections import Counter
from pathlib import Path

import pandas as pd

# Cartelle
RAW_DIR = Path("dataset")          # contiene australian_users_items.json
CLEAN_DIR = Path("dataset_clean")  # contiene steam_games_clean.json
OUTPUT_DIR = Path("analysis_items")
OUTPUT_DIR.mkdir(exist_ok=True)


def stream_users_items(path: Path):
    """
    Legge australian_users_items.json riga per riga (senza caricare tutto in memoria),
    dove ogni riga è un dict Python, per esempio:
    {
      'user_id': '...',
      'items_count': 277,
      'items': [
        {'item_id': '10', 'item_name': 'Counter-Strike', ...},
        ...
      ]
    }
    """
    with open(path, "rb") as f:
        for i, bline in enumerate(f, start=1):
            if not bline.strip():
                continue

            # decodifica robusta
            try:
                line = bline.decode("utf-8", errors="ignore")
            except Exception:
                line = bline.decode("latin-1", errors="ignore")

            line = line.strip()
            if not line:
                continue

            try:
                obj = ast.literal_eval(line)
            except Exception as e:
                print(f"[WARN] Riga {i} non parsabile: {e}")
                continue

            if not isinstance(obj, dict):
                print(f"[WARN] Riga {i} non è un dict, ignorata.")
                continue

            yield obj


def build_interactions_by_game_id(users_items_path: Path) -> pd.DataFrame:
    """
    Conta quante interazioni ha ogni gioco usando l'ID del gioco (item_id).
    Una interazione = un utente possiede quel gioco.
    Ritorna un DataFrame: [item_id, n_interactions].
    """
    counter = Counter()

    for user_obj in stream_users_items(users_items_path):
        items = user_obj.get("items") or []
        for it in items:
            if not isinstance(it, dict):
                continue
            item_id = it.get("item_id")
            if not item_id:
                continue
            item_id = str(item_id).strip()
            counter[item_id] += 1

    df = (
        pd.Series(counter, name="n_interactions")
        .reset_index()
        .rename(columns={"index": "item_id"})
    )
    print(f"Giochi diversi (per ID) trovati nelle interazioni: {len(df)}")
    return df


def main():
    # 1) Costruiamo le interazioni per gioco (ID) a partire dagli utenti
    users_items_path = RAW_DIR / "australian_users_items.json"
    print(f"Carico (in streaming): {users_items_path}")
    interactions_df = build_interactions_by_game_id(users_items_path)

    # 2) Carichiamo i giochi di Steam (clean) con id + title + release_date
    steam_clean_path = CLEAN_DIR / "steam_games_clean.json"
    print(f"Carico giochi clean: {steam_clean_path}")
    steam_df = pd.read_json(steam_clean_path)

    # Controllo che esistano le colonne che ci servono
    required_cols = ["id", "title", "release_date"]
    for col in required_cols:
        if col not in steam_df.columns:
            raise ValueError(
                f"Colonna '{col}' non trovata in steam_games_clean.json. "
                f"Colonne disponibili: {steam_df.columns.tolist()}"
            )

    # 3) Normalizziamo gli ID come stringhe "pulite" per il join
    interactions_df["item_id_norm"] = (
        interactions_df["item_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    steam_df["id_norm"] = (
        steam_df["id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )

    print(f"Giochi in interactions_df: {interactions_df['item_id_norm'].nunique()}")
    print(f"Giochi in steam_clean:     {steam_df['id_norm'].nunique()}")

    # 4) Join tra interazioni e giochi (ID)
    merged = interactions_df.merge(
        steam_df[["id_norm", "title", "release_date"]],
        left_on="item_id_norm",
        right_on="id_norm",
        how="inner",
    )

    print(f"Righe dopo il join (giochi che compaiono in entrambi): {len(merged)}")

    if merged.empty:
        print("\n[ATTENZIONE] Nessun gioco unito tra interazioni e dataset clean.")
        return

    # 5) Estraiamo l'anno di rilascio
    merged["release_year"] = pd.to_datetime(
        merged["release_date"], errors="coerce"
    ).dt.year

    merged = merged.dropna(subset=["release_year"])
    merged["release_year"] = merged["release_year"].astype("Int64")

    # 6) Ordiniamo per anno e, dentro l'anno, per numero di interazioni (desc)
    merged_sorted = (
        merged[["release_year", "item_id", "title", "n_interactions"]]
        .sort_values(["release_year", "n_interactions"], ascending=[True, False])
    )

    # 7) Stampa anno per anno (solo primi N per leggibilità, se vuoi)
    print("\n=== Giochi e interazioni per anno di rilascio ===")
    current_year = None
    for _, row in merged_sorted.iterrows():
        year = int(row["release_year"])
        if current_year != year:
            current_year = year
            print(f"\n--- Anno {year} ---")
        print(f"{row['title']} (ID {row['item_id']}): {row['n_interactions']} interazioni")

    # 8) Salviamo tutto in un CSV unico
    out_path = OUTPUT_DIR / "games_interactions_by_year.csv"
    merged_sorted.to_csv(out_path, index=False)
    print(f"\nCSV con giochi + interazioni per anno salvato in: {out_path.resolve()}")


if __name__ == "__main__":
    main()
