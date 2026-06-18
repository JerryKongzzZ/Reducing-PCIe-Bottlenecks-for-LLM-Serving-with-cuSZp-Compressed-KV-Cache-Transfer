"""
Adaptive PCIe congestion-aware scheduler for KV-cache compression.

This module provides a lightweight scheduler that maps runtime
pending swap volume to a congestion state (GREEN/YELLOW/RED) and
allocates per-layer relative error budgets (eps_rel) accordingly.

The scheduler is intentionally dependency-light and designed to be
instantiated from `compressed_swap.py` at runtime.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CongestionState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class Block:
    # Unique key used by compressed_blocks_store in the patcher: (layer_idx, block_idx)
    key: Tuple[int, int]
    layer_idx: int
    block_idx: int
    size_bytes: int
    tensor: Optional[object] = None


class PCIEAdaptiveScheduler:
    """Simple rule-based scheduler for assigning eps_rel per block.

    - `layer_sensitivity` maps layer index -> one of {'shallow','mid','deep'}
    - thresholds are bytes-in-flight heuristics for green/yellow/red
    - `compute_eps_map` returns a mapping from Block.key -> eps_rel
    """

    def __init__(
        self,
        layer_sensitivity: Dict[int, str],
        low_threshold_bytes: int = 32 * 1024 * 1024,
        high_threshold_bytes: int = 128 * 1024 * 1024,
        policy: Optional[Dict[CongestionState, Dict[str, float]]] = None,
        cuszp_binding: Optional[object] = None,
    ):
        self.layer_sensitivity = layer_sensitivity
        self.low_threshold = int(low_threshold_bytes)
        self.high_threshold = int(high_threshold_bytes)
        self.cuszp_binding = cuszp_binding

        # Default policy: conservative on GREEN, progressively aggressive under RED
        if policy is None:
            self.policy = {
                CongestionState.GREEN: {"shallow": 1e-5, "mid": 1e-5, "deep": 1e-5},
                CongestionState.YELLOW: {"shallow": 1e-5, "mid": 1e-5, "deep": 1e-4},
                CongestionState.RED: {"shallow": 1e-4, "mid": 1e-3, "deep": 1e-2},
            }
        else:
            self.policy = policy

    def aggregate_pending_volume(self, pending_blocks: List[Block]) -> int:
        return sum(b.size_bytes for b in pending_blocks)

    def decide_congestion_state(self, total_bytes: int) -> CongestionState:
        if total_bytes <= self.low_threshold:
            return CongestionState.GREEN
        if total_bytes <= self.high_threshold:
            return CongestionState.YELLOW
        return CongestionState.RED

    def compute_eps_map(self, pending_blocks: List[Block]) -> Dict[Tuple[int, int], float]:
        """Return a mapping from Block.key -> eps_rel (float).

        The scheduler uses the aggregate pending bytes to decide a congestion
        state, and then maps layer categories to eps_rel using `self.policy`.
        """
        total = self.aggregate_pending_volume(pending_blocks)
        state = self.decide_congestion_state(total)
        mapping: Dict[Tuple[int, int], float] = {}

        for b in pending_blocks:
            cat = self.layer_sensitivity.get(b.layer_idx, "deep")
            eps = self.policy[state].get(cat, 1e-4)
            mapping[b.key] = eps

        logger.debug(
            "Computed eps map: state=%s total_bytes=%d assignments=%d",
            state,
            total,
            len(mapping),
        )
        return mapping

    @staticmethod
    def default_layer_sensitivity(num_layers: int) -> Dict[int, str]:
        """Return a reasonable default 3-bin sensitivity mapping.

        We use an approximate split: top 25%% shallow, next 25%% mid, rest deep.
        """
        shallow = max(1, num_layers // 4)
        mid = max(1, num_layers // 4)
        mapping: Dict[int, str] = {}
        for l in range(num_layers):
            if l < shallow:
                mapping[l] = "shallow"
            elif l < shallow + mid:
                mapping[l] = "mid"
            else:
                mapping[l] = "deep"
        return mapping
