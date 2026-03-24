import ast
from pathlib import Path
import pandas as pd

RAW_DIR = Path("dataset")          # dove sta steam_games.json originale
CLEAN_DIR = Path("dataset_clean")  # dove salveremo il nuovo clean
CLEAN_DIR.mkdir(exist_ok=True)

OUT_PATH = CLEAN_DIR / "steam_games_clean_with_id.json"


def stream_steam_games(path: Path):
    """
    Legge steam_games.json riga per riga.
    Ogni riga è un dict Python, tipo:
    {u'publisher': u'Kotoshiro', u'genres': [...], u'id': u'761140', ...}
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


def main():
    raw_path = RAW_DIR / "steam_games.json"
    print(f"Carico (in streaming): {raw_path}")

    records = []

    for i, obj in enumerate(stream_steam_games(raw_path), start=1):
        game_id = obj.get("id") or obj.get("app_id") or obj.get("item_id")
        if game_id is None:
            continue

        record = {
            "id": str(game_id),
            "title": obj.get("title"),
            "release_date": obj.get("release_date"),
            "genres": obj.get("genres"),
            "tags": obj.get("tags"),
            "specs": obj.get("specs"),
            "developer": obj.get("developer"),
            "publisher": obj.get("publisher"),
            "price": obj.get("price"),
            # se vuoi puoi aggiungere anche "discount_price": obj.get("discount_price"),
        }

        records.append(record)

    if not records:
        raise ValueError("Nessun record valido letto da steam_games.json")

    df = pd.DataFrame(records)

    # togli eventuali duplicati per id
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    after = len(df)
    print(f"Giochi totali letti: {before} → dopo drop_duplicates per id: {after}")

    # salviamo in JSON "pulito" con ID
    df.to_json(OUT_PATH, orient="records", indent=2, force_ascii=False)
    print(f"Nuovo dataset clean con ID salvato in: {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
