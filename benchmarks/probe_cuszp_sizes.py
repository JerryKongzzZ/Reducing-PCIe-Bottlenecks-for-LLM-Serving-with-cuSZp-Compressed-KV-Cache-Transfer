"""Probe cuSZp correctness across input sizes including a CPU round trip."""

import argparse
import json
import sys
from pathlib import Path

import torch

PIPELINE_DIR = Path(__file__).resolve().parents[1] / "integration" / "compression_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import cuszp_wrapper_cpp

CUSZP_MODES = {
    "plain": cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
    "fixed": cuszp_wrapper_cpp.CuszpMode.MODE_FIXED,
    "outlier": cuszp_wrapper_cpp.CuszpMode.MODE_OUTLIER,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192, 16384, 20480, 28672, 32768, 36864, 40960, 49152, 65536])
    parser.add_argument("--error-bound", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuszp-mode", choices=tuple(CUSZP_MODES), default="plain")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=args.error_bound,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=CUSZP_MODES[args.cuszp_mode],
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT,
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, 0)
    results = []
    for numel in args.sizes:
        source = torch.randn(numel, dtype=torch.float32, device="cuda")
        capacity = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(
            source.numel() * source.element_size()
        )
        buffer = torch.empty(capacity, dtype=torch.uint8, device="cuda")
        success, buffer, size, actual_eb = compressor.compress(
            source, buffer, args.error_bound
        )
        payload = buffer[: int(size)].cpu().to("cuda")
        restored = torch.empty_like(source)
        decompressed = compressor.decompress(
            payload, int(size), restored, float(actual_eb)
        )
        max_error = float((source - restored).abs().max().item())
        finite = bool(torch.isfinite(restored).all().item())
        within_bound = finite and max_error <= float(actual_eb) * 1.01 + 1e-7
        results.append(
            {
                "numel": numel,
                "success": bool(success and decompressed),
                "finite": finite,
                "within_bound": within_bound,
                "compressed_bytes": int(size),
                "actual_error_bound": float(actual_eb),
                "max_error": max_error,
            }
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
