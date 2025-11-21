import ast
import pandas as pd
from pathlib import Path

RAW_DIR = Path("dataset")
CLEAN_DIR = Path("dataset_clean")

CLEAN_DIR.mkdir(exist_ok=True)

# Colonne che vogliamo tenere
KEEP_COLUMNS = [
    "id",
    "publisher",
    "genres",
    "title",
    "release_date",
    "tags",
    "price",
    "specs",
    "developer",
]

def load_steam_games(path):
    """
    Legge il file steam_games.json dove ogni riga è un dict Python,
    tipo: {u'publisher': u'Kotoshiro', ...}
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = ast.literal_eval(line)  # Converte la stringa in dict Python
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    print(f"Riga {i} non è un dict, ignorata.")
            except Exception as e:
                print(f"Errore alla riga {i}: {e}")
                continue

    if not records:
        raise ValueError("Nessun record valido trovato nel file.")

    return pd.DataFrame(records)


def clean_steam_games():
    input_file = RAW_DIR / "steam_games.json"
    output_file = CLEAN_DIR / "steam_games_clean.json"

    print(f"Leggo il file: {input_file}")
    df = load_steam_games(input_file)

    print("Colonne trovate nel dataset:")
    print(df.columns.tolist())

    # Manteniamo solo le colonne che ci interessano 
    cols_presenti = [c for c in KEEP_COLUMNS if c in df.columns]
    df_clean = df[cols_presenti]

    print("Colonne mantenute:")
    print(df_clean.columns.tolist())

    # Salviamo le informzazioni in un Json pulito 
    df_clean.to_json(output_file, orient="records", indent=2, force_ascii=False)

    print(f"Dataset pulito salvato in: {output_file}")


if __name__ == "__main__":
    clean_steam_games()
