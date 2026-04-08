import json
import glob
import os

# 1. Load Baseline PCIe Speeds
baseline_file = "data/baseline_results.json"
baseline_h2d_bw = 0.0 # Swap IN baseline
baseline_d2h_bw = 0.0 # Swap OUT baseline

if os.path.exists(baseline_file):
    with open(baseline_file, 'r') as f:
        base_data = json.load(f)
        h2d_data = base_data.get("h2d", {})
        d2h_data = base_data.get("d2h", {})
        
        # Get the largest tensor size benchmarked (or exactly 4194304)
        if "bandwidths" in h2d_data and len(h2d_data["bandwidths"]) > 0:
            baseline_h2d_bw = h2d_data["bandwidths"][-1]
        if "bandwidths" in d2h_data and len(d2h_data["bandwidths"]) > 0:
            baseline_d2h_bw = d2h_data["bandwidths"][-1]

result_files = glob.glob("data/*_results.json")
# remove baseline from results
result_files = [rf for rf in result_files if "baseline" not in rf]

summary = []

for rf in result_files:
    with open(rf, 'r') as f:
        data = json.load(f)
    model_name = os.path.basename(rf).replace("_results.json", "")
    
    for res_list in data.get("results", []):
        for res in (res_list if isinstance(res_list, list) else [res_list]):
            if float(res.get("error_bound_setting", 0)) == 1e-4:
                
                orig_size = res.get("original_size_bytes", 0)
                comp_size = res.get("compressed_size_bytes", 0)
                comp_time_ms = res.get("compression_time_ms", 0)
                decomp_time_ms = res.get("decompression_time_ms", 0)
                
                # Baseline latency
                # Swap OUT: Device to Host
                base_swap_out_ms = (orig_size / (baseline_d2h_bw * 1e9)) * 1000 if baseline_d2h_bw > 0 else 0
                # Swap IN: Host to Device
                base_swap_in_ms = (orig_size / (baseline_h2d_bw * 1e9)) * 1000 if baseline_h2d_bw > 0 else 0
                
                # Compressed latency
                # Swap OUT: Compress on GPU + Transfer smaller data to Host
                transfer_out_ms = (comp_size / (baseline_d2h_bw * 1e9)) * 1000 if baseline_d2h_bw > 0 else 0
                eff_swap_out_ms = comp_time_ms + transfer_out_ms
                
                # Swap IN: Transfer smaller data to GPU + Decompress on GPU
                transfer_in_ms = (comp_size / (baseline_h2d_bw * 1e9)) * 1000 if baseline_h2d_bw > 0 else 0
                eff_swap_in_ms = transfer_in_ms + decomp_time_ms
                
                # Effective Bandwidth
                eff_swap_out_bw = (orig_size / (eff_swap_out_ms / 1000)) / 1e9 if eff_swap_out_ms > 0 else 0
                eff_swap_in_bw = (orig_size / (eff_swap_in_ms / 1000)) / 1e9 if eff_swap_in_ms > 0 else 0
                
                # Speedup
                speedup_out = eff_swap_out_bw / baseline_d2h_bw if baseline_d2h_bw > 0 else 0
                speedup_in = eff_swap_in_bw / baseline_h2d_bw if baseline_h2d_bw > 0 else 0
                
                summary.append({
                    "Model": model_name,
                    "Ratio": round(res.get("compression_ratio", 0), 2),
                    "MaxError": f"{res.get('max_absolute_error', 0):.2e}",
                    "Base_Out_BW": f"{baseline_d2h_bw:.2f}",
                    "Eff_Out_BW": f"{eff_swap_out_bw:.2f}",
                    "Out_Speedup": f"{speedup_out:.2f}x",
                    "Base_In_BW": f"{baseline_h2d_bw:.2f}",
                    "Eff_In_BW": f"{eff_swap_in_bw:.2f}",
                    "In_Speedup": f"{speedup_in:.2f}x"
                })

print(f"**Hardware Baseline:**")
print(f"- PCIe Device-to-Host (Swap OUT) Bandwidth: **{baseline_d2h_bw:.2f} GB/s**")
print(f"- PCIe Host-to-Device (Swap IN) Bandwidth: **{baseline_h2d_bw:.2f} GB/s**")
print("")

print("### Compression Metrics vs Baseline")
print("| Model | Ratio | Max Error | Base Swap-Out BW | Eff. Swap-Out BW | Out Speedup | Base Swap-In BW | Eff. Swap-In BW | In Speedup |")
print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
for row in summary:
    print(f"| **{row['Model']}** | `{row['Ratio']}x` | `{row['MaxError']}` | {row['Base_Out_BW']} GB/s | **{row['Eff_Out_BW']} GB/s** | **{row['Out_Speedup']}** | {row['Base_In_BW']} GB/s | **{row['Eff_In_BW']} GB/s** | **{row['In_Speedup']}** |")
print("\n")
