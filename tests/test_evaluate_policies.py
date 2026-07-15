import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_evaluate_policies_falls_back_when_cuszp_missing():
    import benchmarks.evaluate_policies as evaluate_policies

    evaluate_policies.cuszp_wrapper_cpp = None

    tensor = evaluate_policies.torch.randn(16, dtype=evaluate_policies.torch.float32)
    comp_total, comp_time, decomp_time = evaluate_policies.simulate_adaptive_on_flat_tensor(
        tensor,
        compressor=None,
        scheduler_policy={"shallow": 1e-4},
        num_slices=2,
        device_id=0,
    )

    assert comp_total > 0
    assert comp_time >= 0
    assert decomp_time >= 0
