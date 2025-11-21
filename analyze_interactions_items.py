import ast
from collections import Counter
from pathlib import Path

import pandas as pd

RAW_DIR = Path("dataset")
CLEAN_DIR = Path("dataset_clean")
OUTPUT_DIR = Path("analysis_items")

OUTPUT_DIR.mkdir(exist_ok=True)


def stream_users_items(path: Path):
    """
    Legge australian_users_items.json riga per riga (senza caricare tutto in memoria),
    dove ogni riga è un dict Python, per esempio:
    {'user_id': 'Wackky', 'items_count': 0, 'steam_id': '...', 'items': [...]}
    """
    with open(path, "rb") as f:
        for i, bline in enumerate(f, start=1):
            if not bline.strip():
                continue

            # decodifica robusta: UTF-8 con fallback implicito
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

            yield obj  # restituisce un utente alla volta


def main():
    users_items_path = RAW_DIR / "australian_users_items.json"
    print(f"Carico (in streaming): {users_items_path}")

    # contatori
    user_counts = []          # lista di {user_id, n_interactions}
    item_counter = Counter()  # quante volte compare ogni item_id

    n_users = 0

    for user_obj in stream_users_items(users_items_path):
        n_users += 1
        user_id = user_obj.get("user_id")
        items = user_obj.get("items") or []
        items_count = user_obj.get("items_count")

        # se items_count non c'è o non è affidabile, usiamo len(items)
        if items_count is None:
            items_count = len(items)

        # salviamo il numero di interazioni per utente
        user_counts.append(
            {
                "user_id": user_id,
                "n_interactions": items_count,
            }
        )

        # aggiorniamo il conteggio per gioco
        for it in items:
            if isinstance(it, dict):
                item_id = (
                    it.get("item_id")
                    or it.get("id")
                    or it.get("app_id")
                )
            else:
                item_id = it

            if not item_id:
                continue

            item_counter[str(item_id)] += 1

    print(f"Utenti elaborati: {n_users}")
    print(f"Giochi diversi trovati: {len(item_counter)}")

    # DataFrame per utenti
    df_user = pd.DataFrame(user_counts)

    # DataFrame per giochi
    df_game = (
        pd.Series(item_counter, name="n_interactions")
        .reset_index()
        .rename(columns={"index": "item_id"})
    )

    # --- STATISTICHE PER UTENTE ---
    user_stats = df_user["n_interactions"].agg(["count", "mean", "max"])

    print("=== Statistiche (items) per utente ===")
    print(f"Count : {user_stats['count']}")
    print(f"Mean  : {user_stats['mean']:.3f}")
    print(f"Max   : {user_stats['max']}")

    # --- STATISTICHE PER GIOCO ---
    game_stats = df_game["n_interactions"].agg(["count", "mean", "max"])

    print("\n=== Statistiche (items) per gioco ===")
    print(f"Count : {game_stats['count']}")
    print(f"Mean  : {game_stats['mean']:.3f}")
    print(f"Max   : {game_stats['max']}")


    # salvataggio CSV
    df_user.to_csv(OUTPUT_DIR / "items_interactions_per_user.csv", index=False)
    df_game.to_csv(OUTPUT_DIR / "items_interactions_per_game.csv", index=False)

    print(f"\nCSV salvati in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
