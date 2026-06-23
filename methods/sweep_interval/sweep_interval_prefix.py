from __future__ import annotations

from dataclasses import dataclass

from pipeline import lca
from pipeline.prepared import PreparedPoint


@dataclass(frozen=True)
class IntervalSweepIndex:
    """Prepared prefix-count structure used by the interval sweep phase."""

    tip_prefix: list[int]
    tip_preorders: list[int]
    preorder: dict[str, int]
    subtree_end: dict[str, int]


def prepare_interval_sweep_index(prepared_points: list[PreparedPoint]) -> IntervalSweepIndex:
    """Build the prefix-count array needed by the interval sweep.

    This is a preprocessing step. It is intentionally separate from the
    interval classifier so benchmarks can time classification without charging
    this setup cost to the interval method.
    """

    if prepared_points and not isinstance(prepared_points[0], PreparedPoint):
        raise TypeError("prepared_points must contain PreparedPoint objects")

    preorder = lca.precomputed_preorder_f
    subtree_end = lca.precomputed_subtree_end_f
    preorder_nodes = lca.precomputed_preorder_nodes_f
    if preorder is None or subtree_end is None or preorder_nodes is None:
        raise RuntimeError("Call lca.precompute_unique_paths_forward(...) before preparing interval sweep")

    tip_prefix, tip_preorders = _build_tip_arrays(len(preorder_nodes), prepared_points)
    return IntervalSweepIndex(
        tip_prefix=tip_prefix,
        tip_preorders=tip_preorders,
        preorder=preorder,
        subtree_end=subtree_end,
    )


def _build_tip_arrays(node_count: int, prepared_points: list[PreparedPoint]) -> tuple[list[int], list[int]]:
    prefix = [0] * (node_count + 1)
    tip_preorders: list[int] = []
    last_preorder = -1
    sorted_preorders = True

    for point in prepared_points:
        if 0 <= point.preorder_index < node_count:
            prefix[point.preorder_index + 1] += 1
            tip_preorders.append(point.preorder_index)
            if point.preorder_index < last_preorder:
                sorted_preorders = False
            last_preorder = point.preorder_index

    for index in range(1, len(prefix)):
        prefix[index] += prefix[index - 1]

    if not sorted_preorders:
        tip_preorders.sort()

    return prefix, tip_preorders
