import pandas as pd
import gzip
import ast
from tqdm import tqdm

import json

data = json.load(open("/home/marino/tesi/dataset_clean/steam_games_clean.json"))
games_id = set([game['id'] for game in data])
with open("/home/marino/tesi/dataset_clean/steam_games_reviews.json", "w") as f:
    with gzip.open("steam_reviews.json.gz", "rt", encoding="utf-8") as fin:
        for line in tqdm(fin, total=7_793_069):
            try:
                line = ast.literal_eval(line)
                if line['product_id'] in games_id:
                    # Dumps the line as a JSON string
                    f.write(json.dumps(line) + "\n")
            except:
                continue
