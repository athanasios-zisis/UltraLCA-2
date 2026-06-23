import sys
import json
from math import inf

sys.setrecursionlimit(1000000)

from pipeline import lca
from pipeline.lca import precompute_unique_paths_forward
from pipeline.prepared import PreparedSnarlInterval


def _canonicalize_snarls(json_file, graph, root, mapping_base=None, assume_precomputed=False):
    """
    Parse JSON snarls, dedupe them, and orient each pair so the node with
    smaller rooted depth comes first.
    """
    if not assume_precomputed:
        precompute_unique_paths_forward(graph, root)

    snarls = []
    with open(json_file, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or '"directed_acyclic_net_graph"' not in line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "start" not in rec or "end" not in rec:
                continue

            start_side = _json_visit_side(rec["start"], is_start=True, mapping_base=mapping_base)
            end_side = _json_visit_side(rec["end"], is_start=False, mapping_base=mapping_base)
            snarls.append((start_side, end_side))

    canon = []
    seen = set()
    for u, v in snarls:
        if (u, v) in seen:
            continue
        seen.add((u, v))
        du = lca.precomputed_depth_f.get(u, inf)
        dv = lca.precomputed_depth_f.get(v, inf)
        canon.append((u, v) if du <= dv else (v, u))

    return canon


def _json_visit_side(visit, *, is_start, mapping_base=None):
    node_id = visit["node_id"]
    if isinstance(node_id, (list, tuple)):
        node_id = node_id[0]
    if mapping_base:
        node_id = mapping_base.get(node_id, node_id)

    if is_start:
        suffix = "_L" if visit.get("backward", False) else "_R"
    else:
        suffix = "_R" if visit.get("backward", False) else "_L"
    return f"{node_id}{suffix}"


def transform_json_to_snarls(json_file, graph, root, mapping_base=None, assume_precomputed=False):
    """
    Parse DAG snarls from JSON and return the original algorithm's snarl list.
    """
    canon = _canonicalize_snarls(
        json_file,
        graph,
        root,
        mapping_base=mapping_base,
        assume_precomputed=assume_precomputed,
    )

    return sorted(canon, key=_raw_snarl_sort_key)


def _raw_snarl_sort_key(pair):
    return (
        lca.precomputed_depth_f.get(pair[0], inf),
        lca.precomputed_depth_f.get(pair[1], inf),
        pair[0],
        pair[1],
    )


def _build_prepared_snarls_in_preorder(canon):
    preorder = lca.precomputed_preorder_f
    subtree_end = lca.precomputed_subtree_end_f
    preorder_nodes = lca.precomputed_preorder_nodes_f
    if preorder is None or subtree_end is None or preorder_nodes is None:
        raise RuntimeError("Call lca.precompute_unique_paths_forward(...) before preparing sweep snarls")

    buckets = [[] for _ in preorder_nodes]
    for x, y in canon:
        if not (x.endswith("_R") and y.endswith("_L")):
            continue
        left_preorder = preorder[x]
        buckets[left_preorder].append(
            PreparedSnarlInterval(
                left_boundary=x,
                right_boundary=y,
                left_preorder=left_preorder,
                subtree_end_preorder=subtree_end[x],
            )
        )

    return [interval for bucket in buckets for interval in bucket]


def transform_json_to_prepared_snarls(
    json_file,
    graph,
    root,
    mapping_base=None,
    assume_precomputed=False,
):
    """
    Parse JSON snarls and return sweep-ready R-L intervals in linear preorder order.
    """
    canon = _canonicalize_snarls(
        json_file,
        graph,
        root,
        mapping_base=mapping_base,
        assume_precomputed=assume_precomputed,
    )
    return _build_prepared_snarls_in_preorder(canon)


def transform_json_to_snarls_and_prepared(
    json_file,
    graph,
    root,
    mapping_base=None,
    assume_precomputed=False,
    sort_raw_snarls=True,
):
    """
    Parse the JSON once and return both:
      - the original snarl list used by UltraLCA/naive
      - the sweep-ready prepared intervals
    """
    canon = _canonicalize_snarls(
        json_file,
        graph,
        root,
        mapping_base=mapping_base,
        assume_precomputed=assume_precomputed,
    )
    raw_snarls = sorted(canon, key=_raw_snarl_sort_key) if sort_raw_snarls else list(canon)
    prepared_snarls = _build_prepared_snarls_in_preorder(canon)

    return raw_snarls, prepared_snarls
