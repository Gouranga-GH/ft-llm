import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig

# -------------------------------
# 1. Load Pretrained Model & Tokenizer (Step similar to Gemma notebook)
# -------------------------------
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# -------------------------------
# LoRA Configuration (PEFT)
# -------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 2. Prepare a Small Dataset (Step similar to loading quotes in Gemma notebook)
# -------------------------------
# We'll use a small subset of the Rotten Tomatoes dataset (movie reviews)
dataset = load_dataset("rotten_tomatoes", split="train[:100]")  # Only 100 samples for speed

# Save the reviews to a text file for training
train_file = "train_reviews.txt"
with open(train_file, "w", encoding="utf-8") as f:
    for review in dataset["text"]:
        f.write(review.replace("\n", " ") + "\n")

# -------------------------------
# 3. Create TextDataset and DataCollator (Tokenization step)
# -------------------------------
def get_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=64  # Short blocks for quick training
    )

train_dataset = get_dataset(train_file, tokenizer)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# -------------------------------
# 4. Fine-tune the Model (Trainer setup and training, like Gemma notebook)
# -------------------------------
finetuned_model_dir = "finetuned_distilgpt2_lora"
if not os.path.exists(finetuned_model_dir):
    training_args = TrainingArguments(
        output_dir=finetuned_model_dir,      # Where to save model checkpoints and outputs
        overwrite_output_dir=True,           # Overwrite the output directory if it exists
        num_train_epochs=1,                  # Number of times to iterate over the dataset (1 for quick demo)
        per_device_train_batch_size=2,       # Batch size per device (small for low memory usage)
        save_steps=10_000,                   # Save a checkpoint every 10,000 steps (likely only at end for small data)
        save_total_limit=1,                  # Only keep the most recent checkpoint to save disk space
        logging_steps=10,                    # Log training progress every 10 steps
        learning_rate=5e-5,                  # Step size for updating model weights (standard for fine-tuning)
        fp16=False,                          # Use full precision (CPU only; set True for GPU/mixed precision)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(finetuned_model_dir)
    tokenizer.save_pretrained(finetuned_model_dir)
else:
    # Load base model and LoRA adapter
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(model, lora_config)
    model = PeftModel.from_pretrained(model, finetuned_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_dir)

# -------------------------------
# 5. Streamlit UI to Compare Outputs (Before/After fine-tuning)
# -------------------------------
st.title("Movie Review Generator: Before vs After Fine-Tuning")

st.write("""
This app demonstrates the effect of fine-tuning a small language model (DistilGPT-2) on a tiny movie review dataset.
Enter a movie review prompt and compare the model's output before and after fine-tuning.
""")

prompt = st.text_input("Enter the start of a movie review:", "This movie was")

# Function to generate text from a given model and prompt
# Uses sampling for creative, varied outputs
# model: the language model to use (base or fine-tuned)
# tokenizer: tokenizer for encoding/decoding text
# prompt: the initial text to start generation
# max_length: maximum length of the generated sequence
def generate_text(model, tokenizer, prompt, max_length=40):
    # Encode the prompt into input IDs (tokens)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model.generate(
            input_ids,
            max_length=max_length,  # Limit the total length of the output
            num_return_sequences=1,  # Generate only one sequence
            do_sample=True,          # Enable sampling for more diverse outputs
            top_k=50,                # Only sample from the top 50 most likely next tokens
            top_p=0.95,              # Or from the smallest set of tokens with cumulative prob >= 0.95 (nucleus sampling)
            temperature=0.8,         # Controls randomness: lower = more conservative, higher = more random
            pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding if needed
        )
    # Decode the generated tokens back to text, skipping special tokens
    return tokenizer.decode(output[0], skip_special_tokens=True)

if st.button("Generate Review"):
    # Load base model for comparison
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    before = generate_text(base_model, tokenizer, prompt)
    after = generate_text(model, tokenizer, prompt)

    st.subheader("Before Fine-Tuning (Base Model):")
    st.write(before)

    st.subheader("After Fine-Tuning (Fine-Tuned Model):")
    st.write(after)

st.markdown("---")
st.markdown("**References:**")
st.markdown("- [Hugging Face Transformers](https://huggingface.co/transformers/)")
st.markdown("- [Rotten Tomatoes Dataset](https://huggingface.co/datasets/rotten_tomatoes)")

# Clean up temp file
if os.path.exists(train_file):
    os.remove(train_file) 