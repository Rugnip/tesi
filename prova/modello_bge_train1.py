import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    model_name = "unsloth/bge-m3",
    max_seq_length = 256,   # Choose any for long context!
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastSentenceTransformer.get_peft_model(
    model,
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ['key', 'query', 'dense', 'value'],
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

dataset = load_dataset("json", data_files={"train": "/home/marino/tesi/dataset_training/cross_validation_train1.json"})

train_dataset = dataset["train"]

# Rinomina colonne
train_dataset = train_dataset.rename_columns({
    "profilo_utente": "anchor",
    "target": "positive"
})

# Rimuovi colonne inutili
train_dataset = train_dataset.remove_columns([
    "iteration",
    "user_id",
    "prompt"
])

print(train_dataset.column_names)

print("Dataset examples:")
for i in range(6):
    print(train_dataset[i])

from sentence_transformers import util

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
    train_dataset = train_dataset,
    loss = loss,
    args = SentenceTransformerTrainingArguments(
        output_dir = "output",
        num_train_epochs = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,
        learning_rate = 2e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 50,
        warmup_ratio = 0.03,
        report_to = "none",
        lr_scheduler_type = "constant_with_warmup",
        batch_sampler = BatchSamplers.NO_DUPLICATES,
        save_steps=200,
        save_total_limit=2,
    ),

)

trainer_stats = trainer.train()

model.save_pretrained("/home/marino/tesi/models/bge_lora_train1")  # Local saving
model.tokenizer.save_pretrained("/home/marino/tesi/models/bge_lora_train1")