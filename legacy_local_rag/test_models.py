"""
Run this to check if models work before launching the full app.
Usage:
    python test_models.py
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

print("=" * 60)
print("SYSTEM CHECK")
print("=" * 60)
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"VRAM free:  {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

print()

print("=" * 60)
print("TEST 1: QA model Qwen/Qwen2.5-1.5B-Instruct")
print("=" * 60)

try:
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "You answer only from the given evidence."
        },
        {
            "role": "user",
            "content": (
                "Evidence: The Eiffel Tower is located in Paris, France. It was built in 1889.\n\n"
                "Question: Where is the Eiffel Tower located?"
            )
        }
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    pad_id = tok.pad_token_id or tok.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=tok.eos_token_id
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(generated, skip_special_tokens=True).strip()

    print(f"Output: {text}")

    if text:
        print("QA model WORKING")
    else:
        print("QA model generated empty output.")

except Exception as e:
    print(f"QA model FAILED: {type(e).__name__}: {e}")

print()

print("=" * 60)
print("TEST 2: Summary model facebook/bart-large-cnn")
print("=" * 60)

try:
    model_name = "facebook/bart-large-cnn"

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    text = (
        "Machine learning is a method of data analysis that automates analytical model building. "
        "It is based on the idea that systems can learn from data, identify patterns and make decisions "
        "with minimal human intervention."
    )

    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)

    pad_id = tok.pad_token_id or tok.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            min_length=20,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=pad_id
        )

    summary = tok.decode(out[0], skip_special_tokens=True).strip()

    print(f"Output: {summary}")

    if summary:
        print("Summary model WORKING")
    else:
        print("Summary model generated empty output.")

except Exception as e:
    print(f"Summary model FAILED: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("Done.")
print("=" * 60)
