from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreparedPoint:
    """A point from F together with its already-computed preorder index."""

    node_id: str
    preorder_index: int


@dataclass(frozen=True)
class PreparedSnarlInterval:
    """A snarl together with its already-computed subtree sweep interval."""

    left_boundary: str
    right_boundary: str
    left_preorder: int
    subtree_end_preorder: int
