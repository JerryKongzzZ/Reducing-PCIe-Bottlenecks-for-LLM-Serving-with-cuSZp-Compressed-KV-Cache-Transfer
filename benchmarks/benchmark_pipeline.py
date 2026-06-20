import torch
import time
import argparse
import logging
import numpy as np
import json
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.abspath(os.path.join(current_dir, '..', 'integration', 'compression_pipeline'))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

try:
    import cuszp_wrapper_cpp
except ImportError as e:
    logger.error(f"Failed to import cuszp_wrapper_cpp: {e}")
    sys.exit(1)

def generate_kv_cache(model_id, target_size=4194304):
    logger.info(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).cuda()
    
    text = "The Hong Kong Polytechnic University (PolyU) is a public research university located in Hung Hom, Hong Kong. " * 30
    inputs = tokenizer(text, return_tensors="pt", max_length=1000, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    if hasattr(past_key_values, "key_cache"):
        layer_0_key = past_key_values.key_cache[0]
    else:
        layer_0_key = past_key_values[0][0]
    kv_tensor = layer_0_key.to(torch.float32).contiguous().view(-1)
    
    if kv_tensor.numel() < target_size:
        repeats = (target_size // kv_tensor.numel()) + 1
        kv_tensor = kv_tensor.repeat(repeats)[:target_size]
    else:
        kv_tensor = kv_tensor[:target_size]
        
    return kv_tensor.contiguous()

def measure_baseline(tensor, device, num_iterations=50):
    # D2H (Swap OUT Baseline)
    gpu_tensor = tensor.to(device)
    for _ in range(10): _ = gpu_tensor.cpu()
    torch.cuda.synchronize()
    
    d2h_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = gpu_tensor.cpu()
        torch.cuda.synchronize()
        d2h_times.append(time.perf_counter() - start)
        
    avg_d2h = np.mean(d2h_times)
    
    # H2D (Swap IN Baseline)
    cpu_tensor = gpu_tensor.cpu()
    for _ in range(10): _ = cpu_tensor.to(device, non_blocking=False)
    torch.cuda.synchronize()
    
    h2d_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = cpu_tensor.to(device, non_blocking=False)
        torch.cuda.synchronize()
        h2d_times.append(time.perf_counter() - start)
        
    avg_h2d = np.mean(h2d_times)
    
    data_size_bytes = tensor.numel() * 4
    return {
        "d2h_bandwidth": data_size_bytes / avg_d2h / 1e9,
        "h2d_bandwidth": data_size_bytes / avg_h2d / 1e9,
        "d2h_time": avg_d2h,
        "h2d_time": avg_h2d
    }

def measure_compression(tensor, device_id, error_bound=1e-4, num_iterations=50):
    device = torch.device(f"cuda:{device_id}")
    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=error_bound,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, device_id)
    
    estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
        tensor.numel() * 4
    )
    compressed_buffer = torch.empty(estimated_size, dtype=torch.uint8, device=device)
    
    # Warm up
    for _ in range(5):
        success, compressed_buffer, actual_size, actual_eb = compressor.compress(tensor, compressed_buffer)
        if success:
            decomp = torch.empty_like(tensor)
            compressor.decompress(compressed_buffer, actual_size, decomp, actual_eb)
    torch.cuda.synchronize()
    
    comp_times = []
    comp_sizes = []
    actual_eb_last = 0.0
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        success, compressed_buffer, actual_size, actual_eb = compressor.compress(tensor, compressed_buffer)
        torch.cuda.synchronize()
        if success:
            comp_times.append(time.perf_counter() - start)
            comp_sizes.append(actual_size)
            actual_eb_last = actual_eb
            
    decomp_times = []
    if comp_sizes:
        actual_size = comp_sizes[0]
        for _ in range(num_iterations):
            decomp = torch.empty_like(tensor)
            torch.cuda.synchronize()
            start = time.perf_counter()
            compressor.decompress(compressed_buffer, actual_size, decomp, actual_eb_last)
            torch.cuda.synchronize()
            decomp_times.append(time.perf_counter() - start)
            
    decomp = torch.empty_like(tensor)
    compressor.decompress(compressed_buffer, int(np.mean(comp_sizes)), decomp, actual_eb_last)
    abs_errors = torch.abs(tensor - decomp)
    max_error = torch.max(abs_errors).item()
    
    return {
        "comp_time": np.mean(comp_times),
        "decomp_time": np.mean(decomp_times),
        "comp_size": np.mean(comp_sizes),
        "max_error": max_error
    }

def main():
    parser = argparse.ArgumentParser(description="Unified cuSZp Benchmark Pipeline")
    parser.add_argument("--tensor-size", type=int, default=4194304, help="Tensor size to benchmark")
    parser.add_argument("--error-bound", type=float, default=1e-4, help="Relative error bound")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    args = parser.parse_args()

    models = [
        "gpt2",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    ]
    
    device_id = 0
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    
    results = []
    
    for model_id in models:
        logger.info(f"\n==========================================")
        logger.info(f"🧪 Testing: {model_id}")
        logger.info(f"==========================================")
        
        # 1. Generate KV Cache
        tensor = generate_kv_cache(model_id, target_size=args.tensor_size).to(device)
        orig_size_bytes = tensor.numel() * 4
        
        # 2. Baseline Profiling
        base_perf = measure_baseline(tensor, device, num_iterations=args.iterations)
        
        # 3. Compression Profiling
        comp_perf = measure_compression(tensor, device_id, error_bound=args.error_bound, num_iterations=args.iterations)
        
        # 4. Calculate metrics
        ratio = orig_size_bytes / comp_perf["comp_size"]
        
        # Effective Swap OUT (Compress + Transfer smaller data)
        transfer_out_time = comp_perf["comp_size"] / (base_perf["d2h_bandwidth"] * 1e9)
        eff_out_time = comp_perf["comp_time"] + transfer_out_time
        eff_out_bw = (orig_size_bytes / eff_out_time) / 1e9
        
        # Effective Swap IN (Transfer smaller data + Decompress)
        transfer_in_time = comp_perf["comp_size"] / (base_perf["h2d_bandwidth"] * 1e9)
        eff_in_time = transfer_in_time + comp_perf["decomp_time"]
        eff_in_bw = (orig_size_bytes / eff_in_time) / 1e9
        
        out_speedup = eff_out_bw / base_perf["d2h_bandwidth"]
        in_speedup = eff_in_bw / base_perf["h2d_bandwidth"]
        
        safe_name = model_id.replace('/', '_')
        res = {
            "Model": safe_name,
            "Ratio": ratio,
            "MaxError": comp_perf["max_error"],
            "Base_Out_BW": base_perf["d2h_bandwidth"],
            "Eff_Out_BW": eff_out_bw,
            "Out_Speedup": out_speedup,
            "Base_In_BW": base_perf["h2d_bandwidth"],
            "Eff_In_BW": eff_in_bw,
            "In_Speedup": in_speedup
        }
        results.append(res)
        
        # Save KV cache for reference
        os.makedirs("data", exist_ok=True)
        torch.save(tensor.cpu(), f"data/{safe_name}_kv_cache.pt")
        
    # Write summary
    summary_md = "### Compression Metrics vs Baseline\n"
    summary_md += "| Model | Ratio | Max Error | Base Swap-Out | Eff. Swap-Out | Out Speedup | Base Swap-In | Eff. Swap-In | In Speedup |\n"
    summary_md += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for r in results:
        summary_md += f"| **{r['Model']}** | `{r['Ratio']:.2f}x` | `{r['MaxError']:.2e}` | {r['Base_Out_BW']:.2f} GB/s | **{r['Eff_Out_BW']:.2f} GB/s** | **{r['Out_Speedup']:.2f}x** | {r['Base_In_BW']:.2f} GB/s | **{r['Eff_In_BW']:.2f} GB/s** | **{r['In_Speedup']:.2f}x** |\n"
        
    with open("data/benchmark_summary.md", "w") as f:
        f.write(summary_md)
        
    print("\n" + summary_md)

if __name__ == "__main__":
    main()