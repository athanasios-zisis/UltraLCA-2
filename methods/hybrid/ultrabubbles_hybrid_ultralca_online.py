from __future__ import annotations

from dataclasses import dataclass

from pipeline import lca
from pipeline.common import pair_custom_key
from methods.hybrid.ultrabubbles_hybrid_common import (
    Snarl,
    bounded_naive_decision,
    is_rl_snarl,
    validate_alpha_rule,
    validate_prepared_snarls_argument,
)


@dataclass(frozen=True)
class BoundedHybridUltraLCAStats:
    """Summary of one flat online bounded Hybrid(naive, UltraLCA) run."""

    total_snarls: int
    accepted_snarls: int
    rejected_snarls: int
    work_limit: float
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


def find_ultrabubbles_bounded_hybrid_naive_ultralca_online(
    graph,
    snarls: list[Snarl],
    ftip,
    prepared_snarls,
    *,
    alpha: float,
    tips_count: int,
    return_stats: bool = False,
) -> list[Snarl] | tuple[list[Snarl], BoundedHybridUltraLCAStats]:
    """Run flat Hybrid(naive, UltraLCA) with online bounded exact-size counting.

    For each prepared R/L snarl, the function counts original nodes plus
    original edges only until the value exceeds ``alpha * tips_count``. Snarls
    within the bound use the completed naive decision; snarls exceeding the
    bound use the exact UltraLCA ftip/LCA predicate.
    """

    validate_prepared_snarls_argument(
        prepared_snarls,
        "find_ultrabubbles_bounded_hybrid_naive_ultralca_online",
        "fourth",
    )
    validate_alpha_rule(alpha, tips_count=tips_count)

    del snarls  # Raw snarls are kept in the signature for flat-hybrid API symmetry.

    tips = list(ftip)
    work_limit = alpha * tips_count

    accepted: list[Snarl] = []
    total_snarls = 0
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

    for interval in prepared_snarls:
        pair = (interval.left_boundary, interval.right_boundary)
        if not is_rl_snarl(pair):
            continue

        total_snarls += 1
        bounded_naive_attempted_snarls += 1
        bounded = bounded_naive_decision(
            graph,
            interval.left_boundary,
            interval.right_boundary,
            work_limit=work_limit,
        )
        bounded_work_units += bounded.work_units

        if bounded.exceeded:
            bounded_naive_exceeded_snarls += 1
            scan = _find_rejecting_tip(tips, interval.left_boundary, interval.right_boundary)
            ftip_scanned_snarls += 1
            ftip_point_tests += scan.point_tests
            lca_queries += scan.lca_queries
            if scan.witness is None:
                accepted.append(pair)
                algo_fallback_accepted_snarls += 1
            else:
                algo_fallback_rejected_snarls += 1
            continue

        bounded_naive_completed_snarls += 1
        if bounded.is_ultrabubble:
            accepted.append(pair)
            bounded_naive_accepted_snarls += 1
        else:
            bounded_naive_rejected_snarls += 1

    accepted = sorted(set(accepted), key=pair_custom_key)
    stats = BoundedHybridUltraLCAStats(
        total_snarls=total_snarls,
        accepted_snarls=len(accepted),
        rejected_snarls=total_snarls - len(accepted),
        work_limit=work_limit,
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


def _find_rejecting_tip(tips: list[str], x: str, y: str) -> _TipScanResult:
    point_tests = 0
    lca_queries = 0
    for tip in tips:
        point_tests += 1
        left_lca = lca.find_lca_forward(tip, x)
        lca_queries += 1
        if left_lca != x:
            continue
        right_lca = lca.find_lca_forward(tip, y)
        lca_queries += 1
        if right_lca != y:
            return _TipScanResult(tip, point_tests, lca_queries)
    return _TipScanResult(None, point_tests, lca_queries)
