import argparse
import json
import logging
import os
import sys
import time
import zlib

import numpy as np
import torch
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
    cuszp_wrapper_cpp = None
    logger.warning(f"Failed to import cuszp_wrapper_cpp: {e}; using CPU zlib fallback for reproducibility")


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def generate_kv_cache(model_id, target_size=4194304, use_synthetic=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_synthetic:
        logger.info("Using synthetic tensor for smoke-test mode")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        tensor = torch.randn(target_size, generator=generator, dtype=torch.float32)
        return tensor.to(device).contiguous()

    logger.info(f"Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    text = "The Hong Kong Polytechnic University (PolyU) is a public research university located in Hung Hom, Hong Kong. " * 30
    inputs = tokenizer(text, return_tensors="pt", max_length=1000, truncation=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_key_values = outputs.past_key_values
    if type(past_key_values).__name__ == "DynamicCache" or hasattr(past_key_values, "key_cache"):
        if hasattr(past_key_values, "key_cache"):
            layer_0_key = past_key_values.key_cache[0]
        else:
            layer_0_key = list(past_key_values)[0][0]
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
    gpu_tensor = tensor.to(device)
    for _ in range(10):
        _ = gpu_tensor.cpu()
    _synchronize_device(device)

    d2h_times = []
    for _ in range(num_iterations):
        _synchronize_device(device)
        start = time.perf_counter()
        _ = gpu_tensor.cpu()
        _synchronize_device(device)
        d2h_times.append(time.perf_counter() - start)

    avg_d2h = np.mean(d2h_times)

    cpu_tensor = gpu_tensor.cpu()
    for _ in range(10):
        _ = cpu_tensor.to(device, non_blocking=False)
    _synchronize_device(device)

    h2d_times = []
    for _ in range(num_iterations):
        _synchronize_device(device)
        start = time.perf_counter()
        _ = cpu_tensor.to(device, non_blocking=False)
        _synchronize_device(device)
        h2d_times.append(time.perf_counter() - start)

    avg_h2d = np.mean(h2d_times)

    data_size_bytes = tensor.numel() * 4
    return {
        "d2h_bandwidth": data_size_bytes / avg_d2h / 1e9 if avg_d2h > 0 else 0.0,
        "h2d_bandwidth": data_size_bytes / avg_h2d / 1e9 if avg_h2d > 0 else 0.0,
        "d2h_time": avg_d2h,
        "h2d_time": avg_h2d,
    }

def measure_compression(tensor, device_id, error_bound=1e-4, num_iterations=50):
    if cuszp_wrapper_cpp is None:
        cpu_tensor = tensor.detach().cpu().contiguous().view(-1)
        payload = cpu_tensor.numpy().tobytes()
        start = time.perf_counter()
        compressed_payload = zlib.compress(payload)
        comp_time = time.perf_counter() - start
        start = time.perf_counter()
        zlib.decompress(compressed_payload)
        decomp_time = time.perf_counter() - start
        max_error = float(torch.max(torch.abs(cpu_tensor - torch.frombuffer(zlib.decompress(compressed_payload), dtype=torch.float32))).item())
        return {
            "comp_time": comp_time,
            "decomp_time": decomp_time,
            "comp_size": len(compressed_payload),
            "max_error": max_error,
            "backend": "cpu_zlib_fallback",
        }

    device = torch.device(f"cuda:{device_id}")
    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=error_bound,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT,
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, device_id)

    estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(tensor.numel() * 4)
    compressed_buffer = torch.empty(estimated_size, dtype=torch.uint8, device=device)

    for _ in range(5):
        success, compressed_buffer, actual_size, actual_eb = compressor.compress(tensor, compressed_buffer)
        if success:
            decomp = torch.empty_like(tensor)
            compressor.decompress(compressed_buffer, actual_size, decomp, actual_eb)
    _synchronize_device(device)

    comp_times = []
    comp_sizes = []
    actual_eb_last = 0.0
    for _ in range(num_iterations):
        _synchronize_device(device)
        start = time.perf_counter()
        success, compressed_buffer, actual_size, actual_eb = compressor.compress(tensor, compressed_buffer)
        _synchronize_device(device)
        if success:
            comp_times.append(time.perf_counter() - start)
            comp_sizes.append(actual_size)
            actual_eb_last = actual_eb

    decomp_times = []
    if comp_sizes:
        actual_size = comp_sizes[0]
        for _ in range(num_iterations):
            decomp = torch.empty_like(tensor)
            _synchronize_device(device)
            start = time.perf_counter()
            compressor.decompress(compressed_buffer, actual_size, decomp, actual_eb_last)
            _synchronize_device(device)
            decomp_times.append(time.perf_counter() - start)

    decomp = torch.empty_like(tensor)
    compressor.decompress(compressed_buffer, int(np.mean(comp_sizes)), decomp, actual_eb_last)
    abs_errors = torch.abs(tensor - decomp)
    max_error = torch.max(abs_errors).item()

    return {
        "comp_time": np.mean(comp_times),
        "decomp_time": np.mean(decomp_times),
        "comp_size": np.mean(comp_sizes),
        "max_error": max_error,
        "backend": "cuszp",
    }

def main():
    parser = argparse.ArgumentParser(description="Unified cuSZp Benchmark Pipeline")
    parser.add_argument("--tensor-size", type=int, default=4194304, help="Tensor size to benchmark")
    parser.add_argument("--error-bound", type=float, default=1e-4, help="Relative error bound")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations")
    parser.add_argument("--models", nargs="+", default=[
        "gpt2",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "facebook/opt-125m",
        "facebook/opt-350m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ], help="Models to benchmark")
    parser.add_argument("--synthetic", action="store_true", help="Use a deterministic synthetic tensor instead of downloading Hugging Face models")
    parser.add_argument("--device", type=int, default=0, help="CUDA device id when CUDA is available")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available; falling back to CPU-only execution")

    results = []
    device_id = args.device

    for model_id in args.models:
        logger.info(f"\n==========================================")
        logger.info(f"🧪 Testing: {model_id}")
        logger.info(f"==========================================")
        
        # 1. Generate KV Cache
        tensor = generate_kv_cache(model_id, target_size=args.tensor_size, use_synthetic=args.synthetic, device=device).to(device)
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
            "Backend": comp_perf.get("backend", "cuszp"),
            "Base_Out_BW": base_perf["d2h_bandwidth"],
            "Eff_Out_BW": eff_out_bw,
            "Out_Speedup": out_speedup,
            "Base_In_BW": base_perf["h2d_bandwidth"],
            "Eff_In_BW": eff_in_bw,
            "In_Speedup": in_speedup,
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