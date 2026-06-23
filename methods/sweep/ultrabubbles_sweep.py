from __future__ import annotations

# Pure left-to-right sweep over already-prepared arrays.
#
# Upstream preprocessing is assumed to be done already. In particular:
# - snarls are already in sweep order
# - points are already in preorder order
# - every snarl already carries the subtree interval of its left boundary
#
# This module therefore keeps only the sweep phase:
# - advance one global point pointer past points that are permanently irrelevant
# - accept immediately when the first still-relevant point lies beyond the
#   current subtree interval
# - otherwise apply the exact existing LCA rejection predicate on the active
#   points inside that subtree interval

from pipeline import lca
from pipeline.prepared import PreparedPoint, PreparedSnarlInterval


Snarl = tuple[str, str]


def find_ultrabubbles_left_to_right(
    prepared_snarls: list[PreparedSnarlInterval],
    prepared_points: list[PreparedPoint],
) -> list[Snarl]:
    """Run the pure left-to-right sweep over already-prepared arrays.

    The function performs no preprocessing. It assumes:
    - ``prepared_snarls`` is already sorted in sweep order
    - ``prepared_points`` is already sorted by preorder
    - the preorder and subtree-end values are already correct

    Returns:
        Accepted snarls in the original sweep order.
    """

    if prepared_snarls and not isinstance(prepared_snarls[0], PreparedSnarlInterval):
        raise TypeError(
            "find_ultrabubbles_left_to_right now expects a list of "
            "PreparedSnarlInterval objects as its first argument"
        )

    if prepared_points and not isinstance(prepared_points[0], PreparedPoint):
        raise TypeError(
            "find_ultrabubbles_left_to_right now expects a list of "
            "PreparedPoint objects as its second argument"
        )

    if not prepared_snarls:
        return []

    if not prepared_points:
        return [
            (interval.left_boundary, interval.right_boundary)
            for interval in prepared_snarls
        ]

    accepted: list[Snarl] = []
    point_index = 0
    point_count = len(prepared_points)

    for interval_index, interval in enumerate(prepared_snarls):
        # Permanently discard points that lie strictly before the current left
        # boundary. Because later snarls start even further right, these points
        # can never become relevant again.
        while (
            point_index < point_count
            and prepared_points[point_index].preorder_index < interval.left_preorder
        ):
            point_index += 1

        # Once every point is irrelevant, all remaining snarls succeed
        # immediately and stay in sweep order.
        if point_index >= point_count:
            accepted.extend(
                (remaining.left_boundary, remaining.right_boundary)
                for remaining in prepared_snarls[interval_index:]
            )
            break

        # If the first still-relevant point is already to the right of the
        # current subtree interval, no point can reject this snarl.
        if prepared_points[point_index].preorder_index > interval.subtree_end_preorder:
            accepted.append((interval.left_boundary, interval.right_boundary))
            continue

        if not _has_rejecting_point(interval, prepared_points, point_index):
            accepted.append((interval.left_boundary, interval.right_boundary))

    return accepted


def _has_rejecting_point(
    interval: PreparedSnarlInterval,
    prepared_points: list[PreparedPoint],
    start_index: int,
) -> bool:
    """Return ``True`` if an active point triggers the exact LCA rejection rule."""

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
