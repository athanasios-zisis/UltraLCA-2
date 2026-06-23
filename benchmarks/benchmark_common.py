from __future__ import annotations

import io
import json
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from methods.nested.nestedsnarls_dfs_common import (  # noqa: E402
    DfsFamilySummary,
    DfsNestedSnarlNode,
    DfsOrderResult,
    build_dfs_frontier_entry_linear_snarl_forest,
    compute_allowed_dfs_order,
)
from methods.sweep_interval.sweep_interval_prefix import (  # noqa: E402
    IntervalSweepIndex,
    prepare_interval_sweep_index,
)
from pipeline import lca  # noqa: E402
from pipeline.ftip import find_leaf_nodes  # noqa: E402
from pipeline.lca import precompute_unique_paths_forward  # noqa: E402
from pipeline.parse_root import find_root, parse_and_build_bi_graph  # noqa: E402
from pipeline.prepared import PreparedPoint, PreparedSnarlInterval  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "GFA_JSON"
DEFAULT_BENCHMARK_DIR = REPO_ROOT / "benchmarks"
DEFAULT_ALPHA = 1.7

ALL_METHOD_STEMS = [
    "naive",
    "ultralca",
    "sweep",
    "sweep_interval",
    "hybrid_ultralca_online",
    "hybrid_sweep_online",
    "nested_ultralca",
    "nested_hybrid_ultralca",
]

NON_NAIVE_METHOD_STEMS = [method for method in ALL_METHOD_STEMS if method != "naive"]


@dataclass(frozen=True)
class DatasetPair:
    dataset_name: str
    gfa_file: Path
    json_file: Path


@dataclass(frozen=True)
class SharedArtifacts:
    dataset_name: str
    graph: object
    raw_snarls: list[tuple[str, str]]
    canonical_rl_snarls: list[tuple[str, str]]
    normalized_raw_snarls: list[tuple[str, str]]
    root: str
    ftip: set[str]
    prepared_points: list[PreparedPoint]
    prepared_snarls: list[PreparedSnarlInterval]
    prefix_index: IntervalSweepIndex
    dfs_result: DfsOrderResult
    nested_nodes: list[DfsNestedSnarlNode]
    nested_families: list[DfsFamilySummary]


def configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except ValueError:
            pass


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def alpha_tag(alpha: float) -> str:
    return str(alpha).replace("-", "m").replace(".", "p")


def discover_dataset_pairs(dataset_dir: Path, selected_names: Iterable[str] | None) -> list[DatasetPair]:
    wanted = {name.casefold() for name in selected_names} if selected_names else None
    pairs: list[DatasetPair] = []

    for gfa_file in sorted(dataset_dir.glob("*.gfa"), key=lambda path: path.name.casefold()):
        dataset_name = gfa_file.stem
        if wanted is not None and dataset_name.casefold() not in wanted:
            continue

        json_file = dataset_dir / f"{dataset_name}-T.json"
        if not json_file.exists():
            print(f"Skipping {gfa_file.name}: missing {json_file.name}")
            continue
        pairs.append(DatasetPair(dataset_name, gfa_file, json_file))

    if wanted is not None:
        found = {pair.dataset_name.casefold() for pair in pairs}
        for name in sorted(wanted - found):
            print(f"Requested dataset not found: {name}")

    return pairs


def run_quietly(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def time_call(func, *args, quiet: bool = False, **kwargs):
    start = time.perf_counter()
    result = run_quietly(func, *args, **kwargs) if quiet else func(*args, **kwargs)
    return result, time.perf_counter() - start


def json_visit_side(visit, *, is_start: bool, mapping_base=None) -> str:
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


def parse_json_snarls(json_file: Path, mapping_base=None) -> list[tuple[str, str]]:
    snarls: list[tuple[str, str]] = []
    with open(json_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or '"directed_acyclic_net_graph"' not in line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "start" not in record or "end" not in record:
                continue

            start_side = json_visit_side(record["start"], is_start=True, mapping_base=mapping_base)
            end_side = json_visit_side(record["end"], is_start=False, mapping_base=mapping_base)
            snarls.append((start_side, end_side))
    return snarls


def raw_snarl_sort_key(pair: tuple[str, str]) -> tuple[float, float, str, str]:
    return (
        lca.precomputed_depth_f.get(pair[0], inf),
        lca.precomputed_depth_f.get(pair[1], inf),
        pair[0],
        pair[1],
    )


def normalize_snarls_rl(raw_snarls: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    canonical: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for left_boundary, right_boundary in raw_snarls:
        if (left_boundary, right_boundary) in seen:
            continue
        seen.add((left_boundary, right_boundary))
        left_depth = lca.precomputed_depth_f.get(left_boundary, inf)
        right_depth = lca.precomputed_depth_f.get(right_boundary, inf)
        canonical.append(
            (left_boundary, right_boundary)
            if left_depth <= right_depth
            else (right_boundary, left_boundary)
        )

    normalized_raw = sorted(canonical, key=raw_snarl_sort_key)
    return canonical, normalized_raw


def build_ftip_preorder_points(ftip: set[str]) -> list[PreparedPoint]:
    preorder_nodes = lca.precomputed_preorder_nodes_f
    preorder = lca.precomputed_preorder_f
    if preorder_nodes is None or preorder is None:
        raise RuntimeError("Call precompute_unique_paths_forward(...) before building FTIP preorder points")

    prepared_points: list[PreparedPoint] = []
    for node in preorder_nodes:
        if node in ftip:
            prepared_points.append(PreparedPoint(node_id=node, preorder_index=preorder[node]))
    return prepared_points


def build_rl_preorder_intervals(normalized_snarls: list[tuple[str, str]]) -> list[PreparedSnarlInterval]:
    preorder = lca.precomputed_preorder_f
    subtree_end = lca.precomputed_subtree_end_f
    preorder_nodes = lca.precomputed_preorder_nodes_f
    if preorder is None or subtree_end is None or preorder_nodes is None:
        raise RuntimeError("Call precompute_unique_paths_forward(...) before building R-L preorder intervals")

    buckets: list[list[PreparedSnarlInterval]] = [[] for _ in preorder_nodes]
    for left_boundary, right_boundary in normalized_snarls:
        if not (left_boundary.endswith("_R") and right_boundary.endswith("_L")):
            continue
        left_preorder = preorder[left_boundary]
        buckets[left_preorder].append(
            PreparedSnarlInterval(
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                left_preorder=left_preorder,
                subtree_end_preorder=subtree_end[left_boundary],
            )
        )
    return [interval for bucket in buckets for interval in bucket]


def build_shared_artifacts(pair: DatasetPair) -> SharedArtifacts:
    graph = run_quietly(parse_and_build_bi_graph, pair.gfa_file)
    raw_snarls = parse_json_snarls(pair.json_file)

    root = run_quietly(find_root, graph)
    if root is None:
        raise RuntimeError(f"No forward root reaches all nodes for {pair.dataset_name}")

    run_quietly(precompute_unique_paths_forward, graph, root)
    canonical_rl_snarls, normalized_raw_snarls = normalize_snarls_rl(raw_snarls)

    ftip = find_leaf_nodes(graph)
    prepared_points = build_ftip_preorder_points(ftip)
    prepared_snarls = build_rl_preorder_intervals(normalized_raw_snarls)
    prefix_index = prepare_interval_sweep_index(prepared_points)
    dfs_result = compute_allowed_dfs_order(graph, root)
    nested_nodes, nested_families, _audit = build_dfs_frontier_entry_linear_snarl_forest(
        prepared_snarls,
        dfs_result,
    )

    return SharedArtifacts(
        dataset_name=pair.dataset_name,
        graph=graph,
        raw_snarls=raw_snarls,
        canonical_rl_snarls=canonical_rl_snarls,
        normalized_raw_snarls=normalized_raw_snarls,
        root=root,
        ftip=ftip,
        prepared_points=prepared_points,
        prepared_snarls=prepared_snarls,
        prefix_index=prefix_index,
        dfs_result=dfs_result,
        nested_nodes=nested_nodes,
        nested_families=nested_families,
    )
