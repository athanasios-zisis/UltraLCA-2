import sys
sys.setrecursionlimit(1000000)

from pipeline import lca
from pipeline.prepared import PreparedPoint

#FIND LEAF NODES OF THE BIEDGED GRAPH
def find_leaf_nodes(graph):
    """
    Returns a set of all leaf nodes in the graph (nodes with degree one).
    """
    return {node for node in graph.nodes if graph.degree(node) == 1}


def find_leaf_nodes_and_prepared_points(graph):
    """
    Return both the tip set and the sweep-ready PreparedPoint list.

    The PreparedPoint list follows the preorder already computed in lca.py,
    so no extra sorting pass is needed here.
    """
    preorder_nodes = lca.precomputed_preorder_nodes_f
    preorder = lca.precomputed_preorder_f
    if preorder_nodes is None or preorder is None:
        raise RuntimeError("Call lca.precompute_unique_paths_forward(...) before preparing sweep points")

    tip = set()
    prepared_points = []
    for node in preorder_nodes:
        if graph.degree(node) != 1:
            continue
        tip.add(node)
        prepared_points.append(PreparedPoint(node_id=node, preorder_index=preorder[node]))

    return tip, prepared_points
