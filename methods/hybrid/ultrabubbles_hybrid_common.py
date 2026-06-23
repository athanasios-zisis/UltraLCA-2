from __future__ import annotations

from dataclasses import dataclass

from pipeline.prepared import PreparedSnarlInterval


Snarl = tuple[str, str]


@dataclass(frozen=True)
class BoundedNaiveDecision:
    """Naive decision with early stop on the exact original-size metric."""

    exceeded: bool
    is_ultrabubble: bool
    work_units: int


def validate_prepared_snarls_argument(prepared_snarls, function_name: str, position_label: str) -> None:
    """Raise a clear error when a caller passes raw tuples instead of prepared intervals."""

    if prepared_snarls and not isinstance(prepared_snarls[0], PreparedSnarlInterval):
        raise TypeError(
            f"{function_name} expects PreparedSnarlInterval objects as its {position_label} argument"
        )


def validate_alpha_rule(alpha: float | None, *, tips_count: int | None = None) -> None:
    """Validate one alpha-based online-size rule."""

    if alpha is None:
        raise ValueError("alpha is required for the hybrid alpha rule")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if tips_count is not None and tips_count < 1:
        raise ValueError("tips_count must be at least 1")


def is_rl_snarl(pair: Snarl) -> bool:
    """Return ``True`` when the snarl uses the expected R/L boundary orientation."""

    left_boundary, right_boundary = pair
    return left_boundary.endswith("_R") and right_boundary.endswith("_L")


def bounded_naive_decision(graph, x: str, y: str, *, work_limit: float) -> BoundedNaiveDecision:
    """Return the exact naive decision, or stop when original size exceeds the bound.

    The counted size is distinct original segment nodes plus distinct non-black
    original edges in the pruned snarl component. Counting stops immediately
    after the metric becomes larger than ``work_limit``.
    """

    reached: set[str] = {x}
    original_nodes: set[str] = {_segment_id(x)}
    original_edges: set[tuple[str, str]] = set()
    stack = [x]

    def current_total() -> int:
        return len(original_nodes) + len(original_edges)

    while stack:
        node = stack.pop()

        for neighbor, data in graph[node].items():
            if _is_deleted_boundary_black_edge(node, neighbor, data, x, y):
                continue

            if neighbor not in reached:
                reached.add(neighbor)
                original_nodes.add(_segment_id(neighbor))
                stack.append(neighbor)
                if current_total() > work_limit:
                    return BoundedNaiveDecision(True, False, current_total())

            if data.get("color") != "black":
                original_edges.add(_normalized_edge_id(node, neighbor))
                if current_total() > work_limit:
                    return BoundedNaiveDecision(True, False, current_total())

    if y not in reached:
        return BoundedNaiveDecision(False, False, current_total())

    for node in reached - {x, y}:
        degree_in_pruned = 0
        for neighbor, data in graph[node].items():
            if neighbor not in reached:
                continue
            if _is_deleted_boundary_black_edge(node, neighbor, data, x, y):
                continue
            degree_in_pruned += 1
            if degree_in_pruned > 1:
                break
        if degree_in_pruned == 1:
            return BoundedNaiveDecision(False, False, current_total())

    return BoundedNaiveDecision(False, True, current_total())


def _is_deleted_boundary_black_edge(u: str, v: str, data, x: str, y: str) -> bool:
    return data.get("color") == "black" and bool({u, v} & {x, y})


def _segment_id(side_id: str) -> str:
    return side_id.split("_", 1)[0]


def _normalized_edge_id(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u <= v else (v, u)
