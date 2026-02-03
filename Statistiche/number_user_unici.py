import pandas as pd
from pathlib import Path

BASE_DIR = Path(".")

INPUT = "/home/marino/tesi/analysis_items/user_interaction_rating_with_year.csv"
OUTPUT = "/home/marino/tesi/analysis_items/user_unici_per_year.csv"

def main():
    print(f"Carico file: {INPUT}")
    df = pd.read_csv(INPUT)

    # Controllo colonne
    required = ["user_id", "release_year"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Manca la colonna '{col}'.")

    # Assicuriamoci che l'anno sia numerico
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"] = df["release_year"].astype(int)

    # Raggruppiamo per anno e contiamo utenti unici
    summary = (
        df.groupby("release_year")["user_id"]
          .nunique()
          .reset_index(name="unique_users")
          .sort_values("release_year")
    )

    print("\n=== Utenti unici per anno ===")
    print(summary)

    # Salva risultato
    summary.to_csv(OUTPUT, index=False)
    print(f"\nFile salvato in: {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
