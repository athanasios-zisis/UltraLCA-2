from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pipeline import lca
from pipeline.common import pair_custom_key
from methods.nested.nestedsnarls_dfs_common import DfsFamilySummary, DfsNestedSnarlNode


Snarl = tuple[str, str]


@dataclass(frozen=True)
class NestedUltraLCAStats:
    """Summary of one top-down Nested UltraLCA run."""

    families: int
    total_snarls: int
    accepted_snarls: int
    rejected_snarls: int
    accepted_by_ancestor: int
    accepted_by_scan: int
    rejected_by_ftip_scan: int
    ftip_scanned_snarls: int
    ftip_point_tests: int
    lca_queries: int


def find_ultrabubbles_nested_ultralca(
    nested_nodes: list[DfsNestedSnarlNode],
    families: list[DfsFamilySummary],
    ftip: Iterable[str],
    *,
    return_stats: bool = False,
) -> list[Snarl] | tuple[list[Snarl], NestedUltraLCAStats]:
    """Run UltraLCA in top-down nested-family order.

    The decision predicate is unchanged from flat UltraLCA: a snarl
    ``(x, y)`` is rejected when some tip ``t`` satisfies
    ``LCA(t, x) == x`` and ``LCA(t, y) != y``.

    The optimization is outer-to-inner acceptance propagation. Parents are
    processed before children. When a snarl is accepted, all of its direct
    children are marked so they can later be accepted without repeating the
    full ftip/LCA scan. Siblings are still checked independently.
    """

    nodes_by_id = {node.snarl_id: node for node in nested_nodes}
    tips = list(ftip)

    accepted: list[Snarl] = []
    accepted_from_ancestor: set[int] = set()

    accepted_by_ancestor = 0
    accepted_by_scan = 0
    rejected_by_ftip_scan = 0
    ftip_scanned_snarls = 0
    ftip_point_tests = 0
    lca_queries = 0

    for family in families:
        for node_id in reversed(family.bottom_up_order):
            node = nodes_by_id[node_id]

            if node_id in accepted_from_ancestor:
                accepted.append((node.left_boundary, node.right_boundary))
                accepted_by_ancestor += 1
                accepted_from_ancestor.update(node.children_ids)
                continue

            ftip_scanned_snarls += 1
            rejected = False
            for tip in tips:
                ftip_point_tests += 1
                rejects, query_count = _tip_rejects_snarl(tip, node)
                lca_queries += query_count
                if rejects:
                    rejected_by_ftip_scan += 1
                    rejected = True
                    break

            if rejected:
                continue

            accepted.append((node.left_boundary, node.right_boundary))
            accepted_by_scan += 1
            accepted_from_ancestor.update(node.children_ids)

    accepted = sorted(accepted, key=pair_custom_key)
    stats = NestedUltraLCAStats(
        families=len(families),
        total_snarls=len(nested_nodes),
        accepted_snarls=len(accepted),
        rejected_snarls=len(nested_nodes) - len(accepted),
        accepted_by_ancestor=accepted_by_ancestor,
        accepted_by_scan=accepted_by_scan,
        rejected_by_ftip_scan=rejected_by_ftip_scan,
        ftip_scanned_snarls=ftip_scanned_snarls,
        ftip_point_tests=ftip_point_tests,
        lca_queries=lca_queries,
    )
    return (accepted, stats) if return_stats else accepted


def _tip_rejects_snarl(tip: str, node: DfsNestedSnarlNode) -> tuple[bool, int]:
    """Return whether ``tip`` rejects ``node`` and how many LCA calls were used."""

    if lca.find_lca_forward(tip, node.left_boundary) != node.left_boundary:
        return False, 1
    return lca.find_lca_forward(tip, node.right_boundary) != node.right_boundary, 2
