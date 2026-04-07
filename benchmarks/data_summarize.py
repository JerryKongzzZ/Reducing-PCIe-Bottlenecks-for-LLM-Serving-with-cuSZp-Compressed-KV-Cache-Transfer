import json
import glob
import pandas as pd
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
                    "Ratio (x)": res.get("compression_ratio", 0),
                    "Comp. BW (GB/s)": res.get("compression_bandwidth_GB_s", 0),
                    "Decomp. BW (GB/s)": res.get("decompression_bandwidth_GB_s", 0),
                    "Max Error": res.get('max_absolute_error', 0)
                })

df = pd.DataFrame(summary)
print(df.to_markdown(index=False))
