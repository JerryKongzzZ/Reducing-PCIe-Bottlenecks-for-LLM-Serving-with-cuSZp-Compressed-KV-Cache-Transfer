"""
Plot evaluation summary JSON produced by `evaluate_policies.py`.

Produces PNG figures in `data/figures/`.
"""
import json
import os
import sys
import matplotlib.pyplot as plt

def plot_model(model_id, data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    orig = data['orig_size_bytes']

    policies = ['static_cuszp', 'adaptive_red_sim', 'int8', 'zlib']
    comp_sizes = [data[p]['comp_size'] for p in policies]
    eff_out = [data[p].get('eff_out_bw', 0.0) for p in policies]

    # Compression ratio
    ratios = [orig / s if s > 0 else 0 for s in comp_sizes]
    plt.figure(figsize=(6,4))
    plt.bar(policies, ratios, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
    plt.ylabel('Compression ratio (orig / comp)')
    plt.title(f'{model_id}: Compression ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_id}_compression_ratio.png'))
    plt.close()

    # Effective outbound bandwidth
    plt.figure(figsize=(6,4))
    plt.bar(policies, eff_out, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
    plt.ylabel('Effective out bandwidth (GB/s)')
    plt.title(f'{model_id}: Effective out bandwidth')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_id}_eff_out_bw.png'))
    plt.close()

    # Max error (where present)
    max_errors = [data[p].get('max_error', None) for p in policies]
    if any(x is not None for x in max_errors):
        plt.figure(figsize=(6,4))
        xs = [x if x is not None else 0 for x in max_errors]
        plt.bar(policies, xs, color=['#4C72B0','#55A868','#C44E52','#8172B2'])
        plt.ylabel('Max reconstruction error')
        plt.title(f'{model_id}: Max error')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{model_id}_max_error.png'))
        plt.close()


def main():
    in_file = 'data/eval_summary.json'
    if len(sys.argv) > 1:
        in_file = sys.argv[1]

    with open(in_file, 'r') as fh:
        summary = json.load(fh)

    out_dir = os.path.join('data', 'figures')
    for model_id, data in summary.items():
        plot_model(model_id, data, out_dir)

    print(f'Wrote figures to {out_dir}')


if __name__ == '__main__':
    main()
