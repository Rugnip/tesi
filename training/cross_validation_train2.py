import json
from collections import defaultdict
from google import genai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# ===============================
# 1. CARICAMENTO DATI
# ===============================

with open("/home/marino/tesi/dataset_clean/steam_games_clean.json", "r", encoding="utf-8") as f:
    games_data = json.load(f)

games_by_id = {str(g["id"]): g for g in games_data}

reviews_data = []
with open("/home/marino/tesi/dataset_clean/train2_arricchito.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        reviews_data.append(json.loads(line))

# ===============================
# 2. FUNZIONI DI SUPPORTO
# ===============================

def safe_join(value):
    if isinstance(value, list):
        return ", ".join(value)
    if isinstance(value, str):
        return value
    return "N/A"


def build_contextual_review(review, game):
    return f"""
Game title: {game.get('title', 'N/A')}
Genres: {safe_join(game.get('genres'))}
Tags: {safe_join(game.get('tags'))}
Developer: {game.get('developer', 'N/A')}

User review:
{review.get('text', '')}
""".strip()


def leave_one_out(contexts):
    for i in range(len(contexts)):
        yield i, contexts[:i] + contexts[i+1:], contexts[i]


# ===============================
# 3. RAGGRUPPAMENTO PER UTENTE
# ===============================

user_reviews = defaultdict(list)

for r in reviews_data:
    gid = str(r["product_id"])
    if gid not in games_by_id:
        continue
    user_reviews[r["user_id"]].append(r)

# ===============================
# 4. GEMINI CLIENT
# ===============================

client = genai.Client(
    api_key="AIzaSyCrdcd4LMdv76Ze75AiKMdsAVvvztCuJYU"
)


def generate_user_profile(contextual_reviews):
    joined = "\n\n".join(contextual_reviews)

    prompt = f"""
Summarize the user’s videogame preferences in order to create an accurate user profile description that includes: 

-the preferred genres 
-the preferred gameplay style 
-the key elements the user values

Do NOT use bullet points.
Do NOT use headings.

Contextualized reviews:
{joined}
""".strip()

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip(), prompt

def process_one_iteration(user_id, iteration, input_ctx, held_out):
    profile, prompt_used = generate_user_profile(input_ctx)

    return {
        "iteration": iteration,
        "user_id": user_id,
        "profilo_utente": profile,
        "target": held_out,
        "prompt": prompt_used
    }

# ===============================
# 5. CROSS VALIDATION SU TUTTI GLI UTENTI
# ===============================

OUTPUT_PATH = "/home/marino/tesi/dataset_training/cross_validation_train2.json"

results = []
done_tasks = set()

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
        for r in results:
            done_tasks.add((r["user_id"], r["iteration"]))

print(f"Task già completati: {len(done_tasks)}")


print("\nINIZIO CROSS VALIDATION\n")

# ===============================
# 5. COSTRUZIONE TASK
# ===============================

tasks = []

for user_id, reviews in user_reviews.items():

    seen_games = set()
    unique_reviews = []

    for r in reviews:
        gid = str(r["product_id"])
        if gid in seen_games:
            continue
        seen_games.add(gid)
        unique_reviews.append(r)

    if len(unique_reviews) < 2:
        continue

    contexts = []
    for r in unique_reviews:
        game = games_by_id[str(r["product_id"])]
        contexts.append(build_contextual_review(r, game))

    for idx, input_ctx, held_out in leave_one_out(contexts):
        iteration = idx + 1

        if (user_id, iteration) in done_tasks:
            continue

        tasks.append((user_id, iteration, input_ctx, held_out))

print(f"Task rimanenti da processare: {len(tasks)}")

# ===============================
# 6. ESECUZIONE MULTI-THREAD
# ===============================

MAX_WORKERS = os.cpu_count()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = [
        executor.submit(process_one_iteration, *task)
        for task in tasks
    ]

    for future in tqdm(
    as_completed(futures),
    total=len(futures),
    desc="Cross-validation LOO (multi-thread)"
    ):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print("Errore in un task:", e)


results.sort(key=lambda x: (x["user_id"], x["iteration"]))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Risultati salvati in: {OUTPUT_PATH}")
