import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmarks.plot_summary as plot_summary


def test_summary_plot_is_generated(tmp_path):
    summary = {
        "model-a": {
            "orig_size_bytes": 1000,
            "baseline": {"d2h_bandwidth": 10.0, "h2d_bandwidth": 8.0},
            "static_cuszp": {"comp_size": 300, "comp_time": 0.01, "decomp_time": 0.02, "max_error": 1e-4},
            "adaptive_red_sim": {"comp_size": 200, "comp_time": 0.015, "decomp_time": 0.025, "max_error": 2e-4},
            "int8": {"comp_size": 250, "comp_time": 0.02, "decomp_time": 0.03, "max_error": 3e-4},
            "zlib": {"comp_size": 220, "comp_time": 0.025, "decomp_time": 0.035},
        }
    }

    out_path = plot_summary.make_summary_plot(summary, tmp_path)

    assert out_path is not None
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
