from __future__ import annotations

from dataclasses import dataclass

from pipeline import lca
from pipeline.prepared import PreparedPoint, PreparedSnarlInterval
from methods.hybrid.ultrabubbles_hybrid_common import (
    Snarl,
    bounded_naive_decision,
    validate_alpha_rule,
    validate_prepared_snarls_argument,
)


@dataclass(frozen=True)
class BoundedOnlineSweepHybridRunStats:
    """Summary of branch behavior for bounded online Hybrid(naive, sweep)."""

    naive_branch_snarls: int
    naive_branch_ul: int
    sweep_branch_snarls: int
    sweep_branch_ul: int
    sweep_fastpath_snarls: int
    sweep_checked_snarls: int
    bounded_naive_completed_snarls: int
    bounded_naive_exceeded_snarls: int
    bounded_work_units: int


def find_ultrabubbles_bounded_hybrid_naive_sweep_online(
    graph,
    prepared_snarls,
    prepared_points,
    *,
    alpha: float,
    return_stats: bool = False,
) -> list[Snarl] | tuple[list[Snarl], BoundedOnlineSweepHybridRunStats]:
    """Run online Hybrid(naive, sweep) without the exact size prepass.

    This keeps the sweep fast-path first, then uses bounded online
    original-size counting with the same ``alpha * T_suffix(left)`` rule.
    Completed bounded traversals use the exact naive result; exceeded
    traversals fall back to the local sweep rejection test.
    """

    validate_alpha_rule(alpha)
    validate_prepared_snarls_argument(
        prepared_snarls,
        "find_ultrabubbles_bounded_hybrid_naive_sweep_online",
        "second",
    )
    _validate_prepared_points_argument(
        prepared_points,
        "find_ultrabubbles_bounded_hybrid_naive_sweep_online",
        "third",
    )

    if not prepared_snarls:
        stats = BoundedOnlineSweepHybridRunStats(
            naive_branch_snarls=0,
            naive_branch_ul=0,
            sweep_branch_snarls=0,
            sweep_branch_ul=0,
            sweep_fastpath_snarls=0,
            sweep_checked_snarls=0,
            bounded_naive_completed_snarls=0,
            bounded_naive_exceeded_snarls=0,
            bounded_work_units=0,
        )
        return ([], stats) if return_stats else []

    accepted: list[Snarl] = []
    point_index = 0
    point_count = len(prepared_points)

    naive_branch_snarls = 0
    naive_branch_ul = 0
    sweep_branch_snarls = 0
    sweep_branch_ul = 0
    sweep_fastpath_snarls = 0
    sweep_checked_snarls = 0
    bounded_naive_completed_snarls = 0
    bounded_naive_exceeded_snarls = 0
    bounded_work_units = 0

    for interval_index, interval in enumerate(prepared_snarls):
        while (
            point_index < point_count
            and prepared_points[point_index].preorder_index < interval.left_preorder
        ):
            point_index += 1

        if point_index >= point_count:
            remaining_pairs = [
                (remaining.left_boundary, remaining.right_boundary)
                for remaining in prepared_snarls[interval_index:]
            ]
            accepted.extend(remaining_pairs)
            remaining_count = len(remaining_pairs)
            sweep_branch_snarls += remaining_count
            sweep_branch_ul += remaining_count
            sweep_fastpath_snarls += remaining_count
            break

        pair = (interval.left_boundary, interval.right_boundary)

        if prepared_points[point_index].preorder_index > interval.subtree_end_preorder:
            accepted.append(pair)
            sweep_branch_snarls += 1
            sweep_branch_ul += 1
            sweep_fastpath_snarls += 1
            continue

        bounded = bounded_naive_decision(
            graph,
            interval.left_boundary,
            interval.right_boundary,
            work_limit=alpha * (point_count - point_index),
        )
        bounded_work_units += bounded.work_units
        if not bounded.exceeded:
            bounded_naive_completed_snarls += 1
            naive_branch_snarls += 1
            if bounded.is_ultrabubble:
                accepted.append(pair)
                naive_branch_ul += 1
            continue

        bounded_naive_exceeded_snarls += 1
        sweep_branch_snarls += 1
        sweep_checked_snarls += 1
        if not _has_rejecting_point(interval, prepared_points, point_index):
            accepted.append(pair)
            sweep_branch_ul += 1

    stats = BoundedOnlineSweepHybridRunStats(
        naive_branch_snarls=naive_branch_snarls,
        naive_branch_ul=naive_branch_ul,
        sweep_branch_snarls=sweep_branch_snarls,
        sweep_branch_ul=sweep_branch_ul,
        sweep_fastpath_snarls=sweep_fastpath_snarls,
        sweep_checked_snarls=sweep_checked_snarls,
        bounded_naive_completed_snarls=bounded_naive_completed_snarls,
        bounded_naive_exceeded_snarls=bounded_naive_exceeded_snarls,
        bounded_work_units=bounded_work_units,
    )
    return (accepted, stats) if return_stats else accepted


def _has_rejecting_point(
    interval: PreparedSnarlInterval,
    prepared_points: list[PreparedPoint],
    start_index: int,
) -> bool:
    point_index = start_index
    while (
        point_index < len(prepared_points)
        and prepared_points[point_index].preorder_index <= interval.subtree_end_preorder
    ):
        point_node = prepared_points[point_index].node_id
        if (
            lca.find_lca_forward(point_node, interval.left_boundary) == interval.left_boundary
            and lca.find_lca_forward(point_node, interval.right_boundary) != interval.right_boundary
        ):
            return True
        point_index += 1
    return False


def _validate_prepared_points_argument(prepared_points, function_name: str, position_label: str) -> None:
    if prepared_points and not isinstance(prepared_points[0], PreparedPoint):
        raise TypeError(
            f"{function_name} expects PreparedPoint objects as its {position_label} argument"
        )
