import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
torch._dynamo.disable()

from unsloth import FastSentenceTransformer

fourbit_models = [
    "unsloth/all-MiniLM-L6-v2",
    "unsloth/embeddinggemma-300m",
    "unsloth/Qwen3-Embedding-4B",
    "unsloth/Qwen3-Embedding-0.6B",
    "unsloth/all-mpnet-base-v2",
    "unsloth/gte-modernbert-base",
    "unsloth/bge-m3"

] # More models at https://huggingface.co/unsloth

model = FastSentenceTransformer.from_pretrained(
    model_name = "unsloth/Qwen3-Embedding-0.6B",
    max_seq_length = 256,   # Choose any for long context!
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastSentenceTransformer.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = False, # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    task_type = "FEATURE_EXTRACTION"
)

from datasets import load_dataset
dataset = load_dataset("electroglyph/technical", split = "train")

print("Dataset examples:")
for i in range(6):
    print(dataset[i])

from sentence_transformers import util

def test_inference(model, run_name = "Run"):
    """Test model with a query and candidate sentences"""
    query = "apexification"
    candidates = [
        "a brick left by Yuki",  # Completely unrelated
        "apples are a tasty treat",  # Unrelated, but shares "ap-" prefix
        "the weed whacker uses an engine that runs on a mixture of gas and oil",  # Unrelated
        "a type of cancer treatment that uses drugs to boost the body's immune response",  # Medical context but wrong procedure
        "a plant hormone for regulating stress responses",  # Scientific but unrelated field
        "induces root tip closure in non-vital teeth"  # CORRECT - this is what apexification actually means
    ]

    with torch.inference_mode():
      with torch.autocast(device_type = "cuda", dtype = torch.float32):
          query_emb = model.encode(query, convert_to_tensor = True)
          candidate_embs = model.encode(candidates, convert_to_tensor = True)
    scores = util.cos_sim(query_emb, candidate_embs)[0]

    results = []
    for i, score in enumerate(scores):
        results.append((candidates[i], score.item()))
    results.sort(key = lambda x: x[1], reverse = True)

    print(f"\n--- {run_name} Results for query: '{query}' ---")
    for text, score in results:
        print(f"{score:.4f} | {text}")

test_inference(model, run_name = "Pre-Training")

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from unsloth import is_bf16_supported

# This will use other positives in the same batch as negative examples
loss = losses.MultipleNegativesRankingLoss(model)

trainer = SentenceTransformerTrainer(
    model = model,
    train_dataset = dataset,
    loss = loss,
    args = SentenceTransformerTrainingArguments(
        output_dir = "output",
        num_train_epochs = 2,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 32,
        learning_rate = 3e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 50,
        warmup_ratio = 0.03,
        report_to = "none",
        lr_scheduler_type = "constant_with_warmup",
        # Because we have duplicate anchors in the dataset, we don't want
        # to accidentally use them for negative examples
        batch_sampler = BatchSamplers.NO_DUPLICATES,
    ),

)

trainer_stats = trainer.train()
test_inference(model, run_name = "Post-Training")