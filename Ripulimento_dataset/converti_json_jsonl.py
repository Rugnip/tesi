import json

input_path = "/home/marino/tesi/dataset_training/triplet_training_dataset.json"      # <-- il tuo file attuale
output_path = "/home/marino/tesi/dataset_training/triplet_training_dataset.jsonl"    # <-- nuovo file

with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
    data = json.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Conversione completata: {output_path}")
print(f"Numero esempi: {len(data)}")
