# core/model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen2.5-0.5B: ~1GB on disk, runs on CPU, fast enough to iterate
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"

def load_model(model_name: str = DEFAULT_MODEL):
    print(f"loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"loading model weights...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",       # puts on GPU if available, else CPU
        low_cpu_mem_usage=True,  # streams weights instead of loading all at once
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"model loaded on {device}, dtype={dtype}")
    return model, tokenizer

def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,   # HuggingFace's built-in KV cache  you'll replace this in M2
        )

    # only return the newly generated tokens, not the prompt
    new_ids = output_ids[0][input_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "prompt_tokens": input_len,
        "generated_tokens": len(new_ids),
        "text": generated,
    }
    
    
if __name__ == "__main__":
    model, tokenizer = load_model()
    result = generate(model, tokenizer, "The transformer architecture works by")
    print(result)


