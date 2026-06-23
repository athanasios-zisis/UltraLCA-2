from __future__ import annotations

import csv
import heapq
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Iterable

from pipeline import lca
from pipeline.parse_root import allowed_edge


@dataclass(frozen=True)
class DfsOrderResult:
    """Discovery/closing times from a real allowed-edge DFS."""

    order_name: str
    preorder: dict[str, int]
    postorder: dict[str, int]
    subtree_end_pre: dict[str, int]
    forced_discoveries: list[dict[str, object]]
    unreachable_nodes: list[str]
    constraint_count: int = 0


@dataclass
class DfsNestedSnarlNode:
    """One R-L snarl in a DFS-derived family forest."""

    snarl_id: int
    left_boundary: str
    right_boundary: str
    dfs_pre_left: int
    dfs_post_left: int
    dfs_pre_right: int
    dfs_post_right: int
    lca_pre_left: int | None = None
    lca_pre_right: int | None = None
    parent_id: int | None = None
    family_id: int | None = None
    depth_level: int = 0
    bottom_up_rank: int | None = None
    children_ids: list[int] = field(default_factory=list)

    @property
    def children_count(self) -> int:
        return len(self.children_ids)

    @property
    def is_leaf(self) -> bool:
        return not self.children_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "snarl_id": self.snarl_id,
            "family_id": self.family_id,
            "parent_id": self.parent_id,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "dfs_pre_left": self.dfs_pre_left,
            "dfs_post_left": self.dfs_post_left,
            "dfs_pre_right": self.dfs_pre_right,
            "dfs_post_right": self.dfs_post_right,
            "lca_pre_left": self.lca_pre_left,
            "lca_pre_right": self.lca_pre_right,
            "depth_level": self.depth_level,
            "children_count": self.children_count,
            "is_leaf": self.is_leaf,
            "bottom_up_rank": self.bottom_up_rank,
            "children_ids": self.children_ids,
        }


@dataclass(frozen=True)
class DfsFamilySummary:
    """Summary for one top-level DFS-derived snarl family."""

    family_id: int
    root_id: int
    size: int
    max_depth: int
    bottom_up_order: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "root_id": self.root_id,
            "size": self.size,
            "max_depth": self.max_depth,
            "bottom_up_order": self.bottom_up_order,
        }


def compute_allowed_dfs_order(graph, root: str) -> DfsOrderResult:
    """Run a plain DFS on the original graph using only allowed edge rules."""

    successors_by_node = {node: _allowed_successors(graph, node) for node in graph.nodes()}
    preorder: dict[str, int] = {}
    postorder: dict[str, int] = {}
    subtree_end_pre: dict[str, int] = {}
    next_pre = 0
    next_post = 0

    def discover(node: str, stack: list[list[object]]) -> None:
        nonlocal next_pre
        preorder[node] = next_pre
        next_pre += 1
        stack.append([node, successors_by_node[node], 0])

    stack: list[list[object]] = []
    discover(root, stack)

    while stack:
        frame = stack[-1]
        node = frame[0]
        successors = frame[1]
        index = frame[2]

        if index >= len(successors):
            subtree_end_pre[node] = next_pre - 1
            postorder[node] = next_post
            next_post += 1
            stack.pop()
            continue

        successor = successors[index]
        frame[2] = index + 1
        if successor not in preorder:
            discover(successor, stack)

    unreachable_nodes = sorted(
        [node for node in graph.nodes() if node not in preorder],
        key=_node_key,
    )
    return DfsOrderResult(
        order_name="normal_allowed_dfs",
        preorder=preorder,
        postorder=postorder,
        subtree_end_pre=subtree_end_pre,
        forced_discoveries=[],
        unreachable_nodes=unreachable_nodes,
    )


def compute_modified_allowed_dfs_order(graph, root: str, prepared_snarls) -> DfsOrderResult:
    """Run a snarl-aware DFS that delays right-frontier discovery when possible."""

    successors_by_node = {node: _allowed_successors(graph, node) for node in graph.nodes()}
    required_lefts_by_right: dict[str, set[str]] = defaultdict(set)
    for interval in prepared_snarls:
        required_lefts_by_right[interval.right_boundary].add(interval.left_boundary)

    remaining_requirements = {
        right: len(lefts)
        for right, lefts in required_lefts_by_right.items()
    }
    right_boundaries_by_required_left: dict[str, list[str]] = defaultdict(list)
    for right, lefts in required_lefts_by_right.items():
        for left in lefts:
            right_boundaries_by_required_left[left].append(right)

    preorder: dict[str, int] = {}
    postorder: dict[str, int] = {}
    subtree_end_pre: dict[str, int] = {}
    forced_discoveries: list[dict[str, object]] = []
    blocked_predecessors: dict[str, set[str]] = defaultdict(set)
    ready_blocked_targets: list[tuple[tuple[str, int, str, str], str]] = []
    next_pre = 0
    next_post = 0

    def missing_requirements(node: str) -> list[str]:
        return sorted(required_lefts_by_right.get(node, set()) - preorder.keys(), key=_node_key)

    def can_discover(node: str) -> bool:
        return remaining_requirements.get(node, 0) == 0

    def discover(node: str, stack: list[list[object]]) -> None:
        nonlocal next_pre
        preorder[node] = next_pre
        next_pre += 1

        for right_boundary in right_boundaries_by_required_left.get(node, []):
            remaining_requirements[right_boundary] -= 1
            if (
                remaining_requirements[right_boundary] == 0
                and right_boundary in blocked_predecessors
                and right_boundary not in preorder
            ):
                heapq.heappush(ready_blocked_targets, (_node_key(right_boundary), right_boundary))

        stack.append([node, successors_by_node[node], 0])

    stack: list[list[object]] = []
    discover(root, stack)

    while True:
        while stack:
            frame = stack[-1]
            node = frame[0]
            successors = frame[1]
            index = frame[2]

            if index >= len(successors):
                subtree_end_pre[node] = next_pre - 1
                postorder[node] = next_post
                next_post += 1
                stack.pop()
                continue

            successor = successors[index]
            frame[2] = index + 1
            if successor in preorder:
                continue
            if not can_discover(successor):
                blocked_predecessors[successor].add(node)
                continue
            discover(successor, stack)

        while ready_blocked_targets:
            _key, target = heapq.heappop(ready_blocked_targets)
            if (
                target not in preorder
                and remaining_requirements.get(target, 0) == 0
                and blocked_predecessors.get(target)
            ):
                discover(target, stack)
                break
        if stack:
            continue

        forced_target = _choose_forced_target(blocked_predecessors, preorder)
        if forced_target is not None:
            target = forced_target
            source = sorted(blocked_predecessors[target], key=_node_key)[0]
            forced_discoveries.append(
                {
                    "source": source,
                    "target": target,
                    "missing_left_boundaries": missing_requirements(target),
                }
            )
            discover(target, stack)
            continue

        break

    unreachable_nodes = sorted(
        [node for node in graph.nodes() if node not in preorder],
        key=_node_key,
    )
    return DfsOrderResult(
        order_name="modified_allowed_dfs",
        preorder=preorder,
        postorder=postorder,
        subtree_end_pre=subtree_end_pre,
        forced_discoveries=forced_discoveries,
        unreachable_nodes=unreachable_nodes,
        constraint_count=sum(len(lefts) for lefts in required_lefts_by_right.values()),
    )


def build_dfs_nested_snarl_forest(
    prepared_snarls,
    dfs_result: DfsOrderResult,
) -> tuple[list[DfsNestedSnarlNode], list[DfsFamilySummary], dict[str, object]]:
    """Build R-L snarl families from DFS frontier discovery intervals."""

    preorder = dfs_result.preorder
    postorder = dfs_result.postorder
    lca_preorder = lca.precomputed_preorder_f or {}

    missing_boundaries = []
    nodes: list[DfsNestedSnarlNode] = []
    for interval in prepared_snarls:
        if interval.left_boundary not in preorder or interval.right_boundary not in preorder:
            missing_boundaries.append(
                {
                    "left_boundary": interval.left_boundary,
                    "right_boundary": interval.right_boundary,
                }
            )
            continue

        nodes.append(
            DfsNestedSnarlNode(
                snarl_id=len(nodes),
                left_boundary=interval.left_boundary,
                right_boundary=interval.right_boundary,
                dfs_pre_left=preorder[interval.left_boundary],
                dfs_post_left=postorder[interval.left_boundary],
                dfs_pre_right=preorder[interval.right_boundary],
                dfs_post_right=postorder[interval.right_boundary],
                lca_pre_left=lca_preorder.get(interval.left_boundary),
                lca_pre_right=lca_preorder.get(interval.right_boundary),
            )
        )

    nodes = _order_dfs_nodes_linear(nodes, max(preorder.values(), default=0))

    audit = _build_audit_groups(nodes, missing_boundaries, dfs_result)
    crossing_overlaps: list[dict[str, object]] = []

    stack: list[DfsNestedSnarlNode] = []
    roots: list[int] = []

    for node in nodes:
        if not _has_forward_dfs_frontier(node):
            roots.append(node.snarl_id)
            continue

        while stack and not _strictly_contains(stack[-1], node):
            if _crosses(stack[-1], node):
                crossing_overlaps.append(_relationship_record(stack[-1], node))
            stack.pop()

        if stack:
            parent = stack[-1]
            node.parent_id = parent.snarl_id
            parent.children_ids.append(node.snarl_id)
        else:
            roots.append(node.snarl_id)

        stack.append(node)

    families = _assign_families_and_bottom_up_order(nodes, roots)
    audit["crossing_overlaps"] = crossing_overlaps
    return nodes, families, audit


def build_dfs_entry_subtree_snarl_forest(
    prepared_snarls,
    dfs_result: DfsOrderResult,
) -> tuple[list[DfsNestedSnarlNode], list[DfsFamilySummary], dict[str, object]]:
    """Build families using the DFS subtree of each snarl's left frontier.

    A parent P=(x_R,y_L) contains a child C=(a_R,b_L) when both child
    frontiers are inside the DFS subtree rooted at x_R. This fixes the case
    where y_L is discovered early through one branch while DFS later discovers
    nested child frontiers through another branch.
    """

    nodes, missing_boundaries = _make_dfs_nodes(prepared_snarls, dfs_result)
    audit = _build_audit_groups(nodes, missing_boundaries, dfs_result)
    audit["hierarchy_rule"] = "entry_subtree"
    audit["crossing_overlaps"] = []

    if not nodes:
        return nodes, [], audit

    subtree_end = dfs_result.subtree_end_pre
    max_pre = max(dfs_result.preorder.values(), default=0)
    fenwick = _FenwickBest(max_pre + 2)
    candidates_by_right_end = sorted(
        (
            subtree_end[node.right_boundary],
            node.snarl_id,
        )
        for node in nodes
    )
    candidate_index = 0

    roots: list[int] = []
    for node in nodes:
        while (
            candidate_index < len(candidates_by_right_end)
            and candidates_by_right_end[candidate_index][0] < node.dfs_pre_left
        ):
            _right_end, candidate_id = candidates_by_right_end[candidate_index]
            candidate = nodes[candidate_id]
            fenwick.update(
                max_pre - subtree_end[candidate.left_boundary] + 1,
                (candidate.dfs_pre_left, -candidate.snarl_id, candidate.snarl_id),
            )
            candidate_index += 1

        parent_id = fenwick.query(max_pre - node.dfs_pre_right + 1)
        if parent_id is None:
            roots.append(node.snarl_id)
            continue

        parent = nodes[parent_id]
        node.parent_id = parent.snarl_id
        parent.children_ids.append(node.snarl_id)

    families = _assign_families_and_bottom_up_order(nodes, roots)
    return nodes, families, audit


def build_dfs_frontier_entry_snarl_forest(
    prepared_snarls,
    dfs_result: DfsOrderResult,
) -> tuple[list[DfsNestedSnarlNode], list[DfsFamilySummary], dict[str, object]]:
    """Build families with frontier containment plus entry-subtree fallback.

    The primary rule is the simple frontier discovery interval. For snarls
    left as roots by that rule, we add a fallback parent when the snarl starts
    after a candidate parent's right-boundary subtree but before the candidate
    parent's left-boundary subtree ends.
    """

    nodes, _families_unused, audit = build_dfs_nested_snarl_forest(prepared_snarls, dfs_result)
    subtree_end = dfs_result.subtree_end_pre
    max_pre = max(dfs_result.preorder.values(), default=0)
    fenwick = _FenwickBest(max_pre + 2)
    candidates_by_right_end = sorted(
        (
            subtree_end[node.right_boundary],
            node.snarl_id,
        )
        for node in nodes
    )
    candidate_index = 0
    fallback_added = 0

    for node in nodes:
        while (
            candidate_index < len(candidates_by_right_end)
            and candidates_by_right_end[candidate_index][0] < node.dfs_pre_left
        ):
            _right_end, candidate_id = candidates_by_right_end[candidate_index]
            candidate = nodes[candidate_id]
            fenwick.update(
                max_pre - subtree_end[candidate.left_boundary] + 1,
                (candidate.dfs_pre_left, -candidate.snarl_id, candidate.snarl_id),
            )
            candidate_index += 1

        if node.parent_id is not None:
            continue

        parent_id = fenwick.query(max_pre - node.dfs_pre_right + 1)
        if parent_id is None:
            continue
        node.parent_id = parent_id
        fallback_added += 1

    roots: list[int] = []
    for node in nodes:
        node.children_ids = []
        node.family_id = None
        node.depth_level = 0
        node.bottom_up_rank = None

    for node in nodes:
        if node.parent_id is None:
            roots.append(node.snarl_id)
        else:
            nodes[node.parent_id].children_ids.append(node.snarl_id)

    families = _assign_families_and_bottom_up_order(nodes, roots)
    audit["hierarchy_rule"] = "frontier_then_entry_subtree"
    audit["entry_subtree_fallback_added"] = fallback_added
    return nodes, families, audit


def build_dfs_frontier_entry_linear_snarl_forest(
    prepared_snarls,
    dfs_result: DfsOrderResult,
) -> tuple[list[DfsNestedSnarlNode], list[DfsFamilySummary], dict[str, object]]:
    """Build families with a frontier stack plus linear candidate-list splicing.

    The first pass is the same frontier-interval stack used by
    ``build_dfs_nested_snarl_forest``.  The second pass processes fallback
    parents from inner to outer DFS-left discovery order.  A fallback parent can
    claim an unparented snarl or replace a shallower frontier parent, then the
    child is removed from the fallback candidate list so outer parents cannot
    claim it again.
    """

    nodes, _families_unused, audit = build_dfs_nested_snarl_forest(prepared_snarls, dfs_result)
    subtree_end = dfs_result.subtree_end_pre
    fallback_items = nodes
    fallback_item_count = len(fallback_items)
    fallback_pre_right_monotonic = all(
        fallback_items[index - 1].dfs_pre_right <= fallback_items[index].dfs_pre_right
        for index in range(1, fallback_item_count)
    )

    max_pre = max(dfs_result.preorder.values(), default=0)
    first_item_after_pre = _build_first_root_after_pre(fallback_items, max_pre)
    if fallback_pre_right_monotonic:
        last_item_at_most = _build_last_root_with_pre_right_at_most(fallback_items, max_pre)
    else:
        last_item_at_most = _build_last_root_with_pre_left_at_most(fallback_items, max_pre)

    next_item = list(range(fallback_item_count + 1))

    def find_item_index(index: int) -> int:
        while next_item[index] != index:
            next_item[index] = next_item[next_item[index]]
            index = next_item[index]
        return index

    fallback_added = 0
    fallback_reparented = 0
    assignment_scans = 0
    removed_candidates = 0

    for candidate in reversed(nodes):
        candidate_left_end = subtree_end[candidate.left_boundary]
        candidate_right_end = subtree_end[candidate.right_boundary]
        start_threshold = min(max(candidate.dfs_pre_left, candidate_right_end), max_pre)
        end_threshold = min(candidate_left_end, max_pre)
        start_index = first_item_after_pre[start_threshold]
        end_index = last_item_at_most[end_threshold]
        if start_index >= fallback_item_count or start_index > end_index:
            continue

        item_index = find_item_index(start_index)
        while item_index <= end_index:
            child = fallback_items[item_index]
            assignment_scans += 1

            can_contain = (
                child.snarl_id != candidate.snarl_id
                and child.dfs_pre_right <= candidate_left_end
            )
            improves_parent = False
            if can_contain:
                if child.parent_id is None:
                    improves_parent = True
                    fallback_added += 1
                elif candidate.dfs_pre_left > nodes[child.parent_id].dfs_pre_left:
                    improves_parent = True
                    fallback_reparented += 1

            has_deeper_or_equal_parent = (
                child.parent_id is not None
                and nodes[child.parent_id].dfs_pre_left >= candidate.dfs_pre_left
            )
            if improves_parent:
                child.parent_id = candidate.snarl_id
                next_item[item_index] = find_item_index(item_index + 1)
                removed_candidates += 1
                item_index = next_item[item_index]
                continue
            if has_deeper_or_equal_parent:
                next_item[item_index] = find_item_index(item_index + 1)
                removed_candidates += 1
                item_index = next_item[item_index]
                continue

            item_index = find_item_index(item_index + 1)

    roots_ids: list[int] = []
    for node in nodes:
        node.children_ids = []
        node.family_id = None
        node.depth_level = 0
        node.bottom_up_rank = None

    for node in nodes:
        if node.parent_id is None:
            roots_ids.append(node.snarl_id)
        else:
            nodes[node.parent_id].children_ids.append(node.snarl_id)

    families = _assign_families_and_bottom_up_order(nodes, roots_ids)
    audit["hierarchy_rule"] = "frontier_then_entry_subtree_linear_candidate_list"
    audit["entry_subtree_fallback_added"] = fallback_added
    audit["entry_subtree_fallback_reparented"] = fallback_reparented
    audit["linear_candidate_root_count"] = fallback_item_count
    audit["linear_candidate_root_pre_right_monotonic"] = fallback_pre_right_monotonic
    audit["linear_candidate_assignment_scans"] = assignment_scans
    audit["linear_candidate_removed_count"] = removed_candidates
    return nodes, families, audit


def write_dfs_nested_snarl_json(
    output_path: Path,
    *,
    order_name: str,
    gfa_file: Path,
    json_file: Path,
    root: str,
    graph_build_s: float,
    parser_lca_table_s: float,
    dfs_s: float,
    snarls_parse_s: float,
    hierarchy_s: float,
    vg_snarls_t: int,
    rl_snarls_t: int,
    nodes: list[DfsNestedSnarlNode],
    families: list[DfsFamilySummary],
    audit: dict[str, object],
) -> None:
    hierarchy_rule = audit.get("hierarchy_rule")
    is_entry_subtree = hierarchy_rule == "entry_subtree"
    is_frontier_entry = hierarchy_rule == "frontier_then_entry_subtree"
    is_frontier_entry_linear = hierarchy_rule == "frontier_then_entry_subtree_linear_candidate_list"
    definition = {
        "snarl_type": "R-L only",
        "hierarchy_order": order_name,
        "allowed_edges": "black L->R and gray R->L",
        "parser_lca_table": "built only to match existing JSON snarl canonicalization; not used for DFS hierarchy",
    }
    if is_frontier_entry_linear:
        definition.update(
            {
                "nesting_interval": "primary frontier interval with linear entry-subtree candidate-list fallback",
                "parent_rule": (
                    "first use dfs_pre parent frontier containment; then process candidate "
                    "parents from inner to outer and splice each unclaimed or shallower-parented child when "
                    "dfs_end(parent_right) < dfs_pre(child_left) and "
                    "dfs_pre(child_right) <= dfs_end(parent_left)"
                ),
            }
        )
    elif is_frontier_entry:
        definition.update(
            {
                "nesting_interval": "primary frontier interval with entry-subtree fallback for roots",
                "parent_rule": (
                    "first use dfs_pre parent frontier containment; if no parent, "
                    "allow dfs_end(parent_right) < dfs_pre(child_left) and "
                    "dfs_pre(child_right) <= dfs_end(parent_left)"
                ),
            }
        )
    elif is_entry_subtree:
        definition.update(
            {
                "nesting_interval": "entry-subtree annulus: after right_boundary subtree, before left_boundary subtree ends",
                "parent_rule": (
                    "dfs_end(parent_right) < dfs_pre(child_left) and "
                    "dfs_pre(child_right) <= dfs_end(parent_left); choose nearest parent"
                ),
            }
        )
    else:
        definition.update(
            {
                "nesting_interval": "[dfs_pre(left_boundary), dfs_pre(right_boundary)]",
                "parent_rule": "dfs_pre(parent_left) < dfs_pre(child_left) and dfs_pre(child_right) < dfs_pre(parent_right)",
            }
        )

    payload = {
        "format": order_name,
        "format_version": 1,
        "gfa_file": str(gfa_file),
        "json_file": str(json_file),
        "root": root,
        "definition": definition,
        "counts": {
            "vg_snarls_t": vg_snarls_t,
            "rl_snarls_t": rl_snarls_t,
            "families": len(families),
            "leaf_snarls": sum(1 for node in nodes if node.is_leaf),
            "max_depth": max((family.max_depth for family in families), default=0),
            "crossing_overlaps": len(audit["crossing_overlaps"]),
            "shared_left_boundary_groups": len(audit["shared_left_boundary_groups"]),
            "shared_right_boundary_groups": len(audit["shared_right_boundary_groups"]),
            "identical_dfs_interval_groups": len(audit["identical_dfs_interval_groups"]),
            "dfs_reversed_frontier_snarls": audit["dfs_reversed_frontier_count"],
            "forced_dfs_discoveries": len(audit["forced_dfs_discoveries"]),
            "entry_subtree_fallback_added": audit.get("entry_subtree_fallback_added", 0),
            "unreachable_nodes": audit["unreachable_node_count"],
            "snarls_with_missing_boundaries": len(audit["missing_boundaries"]),
        },
        "timings": {
            "graph_build_s": graph_build_s,
            "parser_lca_table_s": parser_lca_table_s,
            "snarls_parse_s": snarls_parse_s,
            "dfs_s": dfs_s,
            "hierarchy_s": hierarchy_s,
        },
        "families": [family.to_dict() for family in families],
        "snarls": [node.to_dict() for node in nodes],
        "audit": audit,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_dfs_nested_snarl_csv(output_path: Path, nodes: list[DfsNestedSnarlNode]) -> None:
    fieldnames = [
        "snarl_id",
        "family_id",
        "parent_id",
        "left_boundary",
        "right_boundary",
        "dfs_pre_left",
        "dfs_post_left",
        "dfs_pre_right",
        "dfs_post_right",
        "lca_pre_left",
        "lca_pre_right",
        "depth_level",
        "children_count",
        "is_leaf",
        "bottom_up_rank",
        "children_ids",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for node in nodes:
            row = node.to_dict()
            row["children_ids"] = ";".join(str(child_id) for child_id in node.children_ids)
            writer.writerow(row)


def _make_dfs_nodes(
    prepared_snarls,
    dfs_result: DfsOrderResult,
) -> tuple[list[DfsNestedSnarlNode], list[dict[str, object]]]:
    preorder = dfs_result.preorder
    postorder = dfs_result.postorder
    lca_preorder = lca.precomputed_preorder_f or {}

    missing_boundaries = []
    nodes: list[DfsNestedSnarlNode] = []
    for interval in prepared_snarls:
        if interval.left_boundary not in preorder or interval.right_boundary not in preorder:
            missing_boundaries.append(
                {
                    "left_boundary": interval.left_boundary,
                    "right_boundary": interval.right_boundary,
                }
            )
            continue

        nodes.append(
            DfsNestedSnarlNode(
                snarl_id=len(nodes),
                left_boundary=interval.left_boundary,
                right_boundary=interval.right_boundary,
                dfs_pre_left=preorder[interval.left_boundary],
                dfs_post_left=postorder[interval.left_boundary],
                dfs_pre_right=preorder[interval.right_boundary],
                dfs_post_right=postorder[interval.right_boundary],
                lca_pre_left=lca_preorder.get(interval.left_boundary),
                lca_pre_right=lca_preorder.get(interval.right_boundary),
            )
        )

    nodes = _order_dfs_nodes_linear(nodes, max(preorder.values(), default=0))
    return nodes, missing_boundaries


def _order_dfs_nodes_linear(
    nodes: list[DfsNestedSnarlNode],
    max_pre: int,
) -> list[DfsNestedSnarlNode]:
    """Order snarls by DFS frontier times without comparison sorting.

    The frontier stack needs ``dfs_pre_left`` increasing and, within the same
    left time, ``dfs_pre_right`` decreasing.  Both keys are DFS preorder
    integers in ``[0, max_pre]``, so two stable counting passes replace the old
    ``O(s log s)`` comparison sort with ``O(max_pre + s)`` ordering.
    """

    if not nodes:
        return nodes

    ordered = _stable_counting_order(
        nodes,
        max_pre,
        lambda node: max_pre - node.dfs_pre_right,
    )
    ordered = _stable_counting_order(
        ordered,
        max_pre,
        lambda node: node.dfs_pre_left,
    )
    for index, node in enumerate(ordered):
        node.snarl_id = index
    return ordered


def _stable_counting_order(
    nodes: list[DfsNestedSnarlNode],
    max_key: int,
    key_func,
) -> list[DfsNestedSnarlNode]:
    counts = [0] * (max_key + 1)
    for node in nodes:
        key = key_func(node)
        if key < 0 or key > max_key:
            raise ValueError(f"DFS preorder key {key} is outside [0, {max_key}]")
        counts[key] += 1

    total = 0
    for key, count in enumerate(counts):
        counts[key] = total
        total += count

    ordered = [nodes[0]] * len(nodes)
    for node in nodes:
        key = key_func(node)
        position = counts[key]
        ordered[position] = node
        counts[key] = position + 1
    return ordered


class _FenwickBest:
    """Fenwick tree storing the best ``(pre_left, tie, snarl_id)`` tuple."""

    def __init__(self, size: int) -> None:
        self._tree: list[tuple[int, int, int] | None] = [None] * (size + 1)
        self._size = size

    def update(self, index: int, value: tuple[int, int, int]) -> None:
        while index <= self._size:
            current = self._tree[index]
            if current is None or value > current:
                self._tree[index] = value
            index += index & -index

    def query(self, index: int) -> int | None:
        if index <= 0:
            return None
        index = min(index, self._size)
        best: tuple[int, int, int] | None = None
        while index > 0:
            current = self._tree[index]
            if current is not None and (best is None or current > best):
                best = current
            index -= index & -index
        return None if best is None else best[2]


def _build_first_root_after_pre(
    roots: list[DfsNestedSnarlNode],
    max_pre: int,
) -> list[int]:
    """Return first root index whose left frontier is discovered after t."""

    first_after = [len(roots)] * (max_pre + 1)
    root_index = 0
    for preorder in range(max_pre + 1):
        while root_index < len(roots) and roots[root_index].dfs_pre_left <= preorder:
            root_index += 1
        first_after[preorder] = root_index
    return first_after


def _build_last_root_with_pre_right_at_most(
    roots: list[DfsNestedSnarlNode],
    max_pre: int,
) -> list[int]:
    """Return last root index whose right frontier preorder is at most t."""

    last_at_most = [-1] * (max_pre + 1)
    root_index = -1
    for preorder in range(max_pre + 1):
        while (
            root_index + 1 < len(roots)
            and roots[root_index + 1].dfs_pre_right <= preorder
        ):
            root_index += 1
        last_at_most[preorder] = root_index
    return last_at_most


def _build_last_root_with_pre_left_at_most(
    roots: list[DfsNestedSnarlNode],
    max_pre: int,
) -> list[int]:
    """Return last root index whose left frontier preorder is at most t."""

    last_at_most = [-1] * (max_pre + 1)
    root_index = -1
    for preorder in range(max_pre + 1):
        while (
            root_index + 1 < len(roots)
            and roots[root_index + 1].dfs_pre_left <= preorder
        ):
            root_index += 1
        last_at_most[preorder] = root_index
    return last_at_most


def _allowed_successors(graph, node: str) -> list[str]:
    return sorted(
        [
            neighbor
            for neighbor, data in graph[node].items()
            if allowed_edge(node, neighbor, data.get("color"))
        ],
        key=_node_key,
    )


def _choose_forced_target(
    blocked_predecessors: dict[str, set[str]],
    preorder: dict[str, int],
) -> str | None:
    candidates = [
        target
        for target, predecessors in blocked_predecessors.items()
        if target not in preorder and predecessors
    ]
    if not candidates:
        return None
    return min(candidates, key=_node_key)


def _node_key(node: str) -> tuple[str, int, str, str]:
    match = re.match(r"([A-Za-z_.-]*?)(\d+)?(_[LR])?$", node)
    if match is None:
        return (node, inf, "", node)
    prefix, number, suffix = match.groups()
    numeric = int(number) if number is not None else inf
    return (prefix or "", numeric, suffix or "", node)


def _strictly_contains(parent: DfsNestedSnarlNode, child: DfsNestedSnarlNode) -> bool:
    return (
        _has_forward_dfs_frontier(parent)
        and _has_forward_dfs_frontier(child)
        and parent.dfs_pre_left < child.dfs_pre_left
        and child.dfs_pre_right < parent.dfs_pre_right
    )


def _crosses(left: DfsNestedSnarlNode, right: DfsNestedSnarlNode) -> bool:
    return (
        _has_forward_dfs_frontier(left)
        and _has_forward_dfs_frontier(right)
        and left.dfs_pre_left < right.dfs_pre_left < left.dfs_pre_right < right.dfs_pre_right
    )


def _has_forward_dfs_frontier(node: DfsNestedSnarlNode) -> bool:
    return node.dfs_pre_left < node.dfs_pre_right


def _relationship_record(first: DfsNestedSnarlNode, second: DfsNestedSnarlNode) -> dict[str, object]:
    return {
        "first_id": first.snarl_id,
        "first": [first.left_boundary, first.right_boundary],
        "first_dfs_interval": [first.dfs_pre_left, first.dfs_pre_right],
        "second_id": second.snarl_id,
        "second": [second.left_boundary, second.right_boundary],
        "second_dfs_interval": [second.dfs_pre_left, second.dfs_pre_right],
    }


def _build_audit_groups(
    nodes: list[DfsNestedSnarlNode],
    missing_boundaries: list[dict[str, object]],
    dfs_result: DfsOrderResult,
) -> dict[str, object]:
    by_left: defaultdict[str, list[DfsNestedSnarlNode]] = defaultdict(list)
    by_right: defaultdict[str, list[DfsNestedSnarlNode]] = defaultdict(list)
    by_interval: defaultdict[tuple[int, int], list[DfsNestedSnarlNode]] = defaultdict(list)

    for node in nodes:
        by_left[node.left_boundary].append(node)
        by_right[node.right_boundary].append(node)
        by_interval[(node.dfs_pre_left, node.dfs_pre_right)].append(node)

    return {
        "shared_left_boundary_groups": _serialize_groups(by_left.values()),
        "shared_right_boundary_groups": _serialize_groups(by_right.values()),
        "identical_dfs_interval_groups": _serialize_groups(by_interval.values()),
        "dfs_reversed_frontier_count": sum(1 for node in nodes if not _has_forward_dfs_frontier(node)),
        "dfs_reversed_frontier_examples": [
            {
                "snarl_id": node.snarl_id,
                "left_boundary": node.left_boundary,
                "right_boundary": node.right_boundary,
                "dfs_interval": [node.dfs_pre_left, node.dfs_pre_right],
                "lca_interval": [node.lca_pre_left, node.lca_pre_right],
            }
            for node in nodes
            if not _has_forward_dfs_frontier(node)
        ][:20],
        "forced_dfs_discoveries": dfs_result.forced_discoveries[:50],
        "forced_dfs_discovery_count": len(dfs_result.forced_discoveries),
        "unreachable_nodes": dfs_result.unreachable_nodes[:50],
        "unreachable_node_count": len(dfs_result.unreachable_nodes),
        "constraint_count": dfs_result.constraint_count,
        "missing_boundaries": missing_boundaries,
    }


def _serialize_groups(groups: Iterable[list[DfsNestedSnarlNode]]) -> list[dict[str, object]]:
    serialized = []
    for group in groups:
        if len(group) < 2:
            continue
        serialized.append(
            {
                "snarl_ids": [node.snarl_id for node in group],
                "snarls": [
                    {
                        "left_boundary": node.left_boundary,
                        "right_boundary": node.right_boundary,
                        "dfs_interval": [node.dfs_pre_left, node.dfs_pre_right],
                        "left_dfs_node_interval": [node.dfs_pre_left, node.dfs_post_left],
                        "right_dfs_node_interval": [node.dfs_pre_right, node.dfs_post_right],
                    }
                    for node in group
                ],
            }
        )
    return serialized


def _assign_families_and_bottom_up_order(
    nodes: list[DfsNestedSnarlNode],
    roots: list[int],
) -> list[DfsFamilySummary]:
    by_id = {node.snarl_id: node for node in nodes}
    families = []

    for family_id, root_id in enumerate(roots):
        family_nodes: list[int] = []
        bottom_up_order: list[int] = []

        def assign_depth(node_id: int, depth: int) -> None:
            node = by_id[node_id]
            node.family_id = family_id
            node.depth_level = depth
            family_nodes.append(node_id)
            for child_id in node.children_ids:
                assign_depth(child_id, depth + 1)

        def postorder(node_id: int) -> None:
            node = by_id[node_id]
            for child_id in node.children_ids:
                postorder(child_id)
            node.bottom_up_rank = len(bottom_up_order)
            bottom_up_order.append(node_id)

        assign_depth(root_id, 0)
        postorder(root_id)
        families.append(
            DfsFamilySummary(
                family_id=family_id,
                root_id=root_id,
                size=len(family_nodes),
                max_depth=max(by_id[node_id].depth_level for node_id in family_nodes),
                bottom_up_order=bottom_up_order,
            )
        )

    return families
