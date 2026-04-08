import json
import glob
import os

result_files = glob.glob("data/*_results.json")
summary = []

for rf in result_files:
    with open(rf, 'r') as f:
        data = json.load(f)
    model_name = os.path.basename(rf).replace("_results.json", "")
    for res_list in data.get("results", []):
        for res in (res_list if isinstance(res_list, list) else [res_list]):
            if float(res.get("error_bound_setting", 0)) == 1e-4:
                summary.append({
                    "Model": model_name,
                    "Ratio": round(res.get("compression_ratio", 0), 2),
                    "CompBW": round(res.get("compression_bandwidth_GB_s", 0), 2),
                    "DecompBW": round(res.get("decompression_bandwidth_GB_s", 0), 2),
                    "MaxError": f"{res.get('max_absolute_error', 0):.2e}"
                })

print("\n\n| Model | Compression Ratio | Compression Speed (GB/s) | Decompression Speed (GB/s) | Absolute Max Error |")
print("| :--- | :--- | :--- | :--- | :--- |")
for row in summary:
    print(f"| **{row['Model']}** | `{row['Ratio']}x` | `{row['CompBW']}` | `{row['DecompBW']}` | `{row['MaxError']}` |")
print("\n")
