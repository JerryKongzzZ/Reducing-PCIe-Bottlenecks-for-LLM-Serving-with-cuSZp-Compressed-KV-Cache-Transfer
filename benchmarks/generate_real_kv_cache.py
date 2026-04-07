"""
Script to generate and dump real KV cache data using a small HuggingFace model.
"""
import torch
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Generate real KV Cache tensor")
    parser.add_argument("--model", type=str, default="gpt2", help="HuggingFace model ID")
    parser.add_argument("--output", type=str, default="data/real_kv_cache.pt", help="Output file")
    args = parser.parse_args()

    print(f"Loading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    
    print("Generating text to populate KV cache...")
    # Make sure text doesn't exceed GPT-2 max length (1024)
    text = "The Hong Kong Polytechnic University (PolyU) is a public research university located in Hung Hom, Hong Kong. " * 30
    inputs = tokenizer(text, return_tensors="pt", max_length=1000, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    
    # Extract the key cache of the first layer
    layer_0_key = past_key_values[0][0]
    
    # Flatten to 1D for our benchmark purposes
    kv_tensor = layer_0_key.to(torch.float32).contiguous().view(-1)
    
    # Repeat the tensor to make it large enough for benchmarking (at least 1M elements)
    target_size = 1048576 * 4 # roughly 4M elements to be safe
    repeats = (target_size // kv_tensor.numel()) + 1
    kv_tensor = kv_tensor.repeat(repeats)
    
    print(f"Extracted KV Cache shape: {layer_0_key.shape}, Total elements after expanding: {kv_tensor.numel()}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(kv_tensor, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
