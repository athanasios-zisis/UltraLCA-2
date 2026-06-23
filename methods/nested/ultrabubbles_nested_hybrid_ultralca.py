from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline import lca
from pipeline.common import pair_custom_key
from methods.hybrid.ultrabubbles_hybrid_common import bounded_naive_decision, validate_alpha_rule
from methods.nested.nestedsnarls_dfs_common import DfsFamilySummary, DfsNestedSnarlNode


Snarl = tuple[str, str]


@dataclass(frozen=True)
class NestedBoundedHybridUltraLCAStats:
    """Summary of one top-down Nested Bounded Hybrid(naive, UltraLCA) run."""

    families: int
    total_snarls: int
    accepted_snarls: int
    rejected_snarls: int
    work_limit: float
    accepted_by_ancestor: int
    bounded_naive_attempted_snarls: int
    bounded_naive_completed_snarls: int
    bounded_naive_exceeded_snarls: int
    bounded_naive_accepted_snarls: int
    bounded_naive_rejected_snarls: int
    algo_fallback_accepted_snarls: int
    algo_fallback_rejected_snarls: int
    bounded_work_units: int
    ftip_scanned_snarls: int
    ftip_point_tests: int
    lca_queries: int


@dataclass(frozen=True)
class _TipScanResult:
    witness: str | None
    point_tests: int
    lca_queries: int


def find_ultrabubbles_nested_bounded_hybrid_naive_ultralca(
    graph,
    nested_nodes: list[DfsNestedSnarlNode],
    families: list[DfsFamilySummary],
    ftip: Iterable[str],
    *,
    alpha: float,
    tips_count: int,
    return_stats: bool = False,
) -> list[Snarl] | tuple[list[Snarl], NestedBoundedHybridUltraLCAStats]:
    """Run Nested Hybrid with a bounded outer-to-inner naive attempt.

    For every snarl not already accepted by an accepted ancestor, the function
    counts original nodes plus original edges and stops as soon as that value
    exceeds ``alpha * tips_count``. If the bound is exceeded, the snarl falls
    back to the exact UltraLCA ftip/LCA predicate.
    """

    validate_alpha_rule(alpha, tips_count=tips_count)

    nodes_by_id = {node.snarl_id: node for node in nested_nodes}
    tips = list(ftip)
    work_limit = alpha * tips_count

    accepted: list[Snarl] = []
    accepted_from_ancestor: set[int] = set()

    accepted_by_ancestor = 0
    bounded_naive_attempted_snarls = 0
    bounded_naive_completed_snarls = 0
    bounded_naive_exceeded_snarls = 0
    bounded_naive_accepted_snarls = 0
    bounded_naive_rejected_snarls = 0
    algo_fallback_accepted_snarls = 0
    algo_fallback_rejected_snarls = 0
    bounded_work_units = 0
    ftip_scanned_snarls = 0
    ftip_point_tests = 0
    lca_queries = 0

    for family in families:
        for node_id in reversed(family.bottom_up_order):
            node = nodes_by_id[node_id]
            pair = (node.left_boundary, node.right_boundary)

            if node_id in accepted_from_ancestor:
                accepted.append(pair)
                accepted_by_ancestor += 1
                accepted_from_ancestor.update(node.children_ids)
                continue

            bounded_naive_attempted_snarls += 1
            bounded = bounded_naive_decision(
                graph,
                node.left_boundary,
                node.right_boundary,
                work_limit=work_limit,
            )
            bounded_work_units += bounded.work_units

            if bounded.exceeded:
                bounded_naive_exceeded_snarls += 1
                scan = _find_rejecting_tip(tips, node)
                ftip_scanned_snarls += 1
                ftip_point_tests += scan.point_tests
                lca_queries += scan.lca_queries
                if scan.witness is not None:
                    algo_fallback_rejected_snarls += 1
                else:
                    accepted.append(pair)
                    algo_fallback_accepted_snarls += 1
                    accepted_from_ancestor.update(node.children_ids)
                continue

            bounded_naive_completed_snarls += 1
            if bounded.is_ultrabubble:
                accepted.append(pair)
                bounded_naive_accepted_snarls += 1
                accepted_from_ancestor.update(node.children_ids)
                continue

            bounded_naive_rejected_snarls += 1
            scan = _find_rejecting_tip(tips, node)
            ftip_scanned_snarls += 1
            ftip_point_tests += scan.point_tests
            lca_queries += scan.lca_queries
            if scan.witness is not None:
                algo_fallback_rejected_snarls += 1

    accepted = sorted(accepted, key=pair_custom_key)
    stats = NestedBoundedHybridUltraLCAStats(
        families=len(families),
        total_snarls=len(nested_nodes),
        accepted_snarls=len(accepted),
        rejected_snarls=len(nested_nodes) - len(accepted),
        work_limit=work_limit,
        accepted_by_ancestor=accepted_by_ancestor,
        bounded_naive_attempted_snarls=bounded_naive_attempted_snarls,
        bounded_naive_completed_snarls=bounded_naive_completed_snarls,
        bounded_naive_exceeded_snarls=bounded_naive_exceeded_snarls,
        bounded_naive_accepted_snarls=bounded_naive_accepted_snarls,
        bounded_naive_rejected_snarls=bounded_naive_rejected_snarls,
        algo_fallback_accepted_snarls=algo_fallback_accepted_snarls,
        algo_fallback_rejected_snarls=algo_fallback_rejected_snarls,
        bounded_work_units=bounded_work_units,
        ftip_scanned_snarls=ftip_scanned_snarls,
        ftip_point_tests=ftip_point_tests,
        lca_queries=lca_queries,
    )
    return (accepted, stats) if return_stats else accepted


def _find_rejecting_tip(tips: list[str], node) -> _TipScanResult:
    point_tests = 0
    lca_queries = 0
    for tip in tips:
        point_tests += 1
        left_lca = lca.find_lca_forward(tip, node.left_boundary)
        lca_queries += 1
        if left_lca != node.left_boundary:
            continue
        right_lca = lca.find_lca_forward(tip, node.right_boundary)
        lca_queries += 1
        if right_lca != node.right_boundary:
            return _TipScanResult(tip, point_tests, lca_queries)
    return _TipScanResult(None, point_tests, lca_queries)
