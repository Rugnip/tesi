from pathlib import Path
import pandas as pd

# Cartelle
OUTPUT_DIR = Path("analysis_items")  # contiene items_interactions_per_game.csv


def main():
    # 1) Carichiamo il file già pronto con item_id, n_interactions, release_year
    interactions_path = OUTPUT_DIR / "items_interactions_per_game.csv"
    print(f"Carico interazioni per gioco da: {interactions_path}")

    df = pd.read_csv(interactions_path)

    # Controllo colonne
    required_cols = ["item_id", "n_interactions", "release_year"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Colonna '{col}' non trovata in items_interactions_per_game.csv. "
                f"Colonne disponibili: {df.columns.tolist()}"
            )

    # 2) Assicuriamoci che l'anno sia numerico
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    # 3) Sommiamo le interazioni per anno
    summary = (
        df.groupby("release_year")["n_interactions"]
        .sum()
        .reset_index()
        .sort_values("release_year")
    )

    # 4) Stampa in formato "anno, totale_interazioni"
    print("\n=== Totale interazioni per anno ===")
    for _, row in summary.iterrows():
        year = int(row["release_year"]) # type: ignore
        total_inter = int(row["n_interactions"]) # type: ignore
        print(f"{year},{total_inter}")

    # 5) Salviamo il risultato in un CSV riassuntivo
    out_path = OUTPUT_DIR / "statistiche_luglio_2017/interactions_per_year_games_with_users.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nCSV riassuntivo salvato in: {out_path.resolve()}")


if __name__ == "__main__":
    main()

