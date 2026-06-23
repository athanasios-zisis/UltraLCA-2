from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.common import pair_custom_key
from pipeline.ftip import find_leaf_nodes_and_prepared_points
from pipeline.lca import precompute_unique_paths_forward
from pipeline.parse_root import find_root, parse_and_build_bi_graph
from pipeline.prepared import PreparedPoint, PreparedSnarlInterval
from pipeline.snarls_vg import transform_json_to_snarls_and_prepared
from methods.sweep_interval.sweep_interval_prefix import IntervalSweepIndex, prepare_interval_sweep_index
from methods.sweep.ultrabubbles_sweep import find_ultrabubbles_left_to_right


Snarl = tuple[str, str]

HERE = REPO_ROOT
DEFAULT_DATASET_DIR = REPO_ROOT / "GFA_JSON"


def find_ultrabubbles_interval_sweep(
    prepared_snarls: list[PreparedSnarlInterval],
    sweep_index: IntervalSweepIndex,
) -> list[Snarl]:
    """Classify snarls by prefix-counting rejecting tips over LCA-tree intervals.

    This is equivalent to the UltraLCA predicate:

        exists t in F such that LCA(t, x) == x and LCA(t, y) != y

    In the LCA tree, LCA(t, x) == x means t is inside subtree(x), and
    LCA(t, y) == y means t is inside subtree(y).  Therefore a snarl is rejected
    exactly when subtree(x) contains at least one prepared point outside
    subtree(y).

    The function assumes ``prepared_snarls`` is already in sweep order and
    ``sweep_index`` has already been prepared.  It performs no graph traversal,
    no prefix construction, and no snarl sorting.  Sparse tip cases fast-path
    by carrying the same monotone tip pointer as the old sweep, then using
    prefix counts only when a tip falls inside subtree(x).
    """

    _validate_inputs(prepared_snarls, sweep_index)
    if not prepared_snarls:
        return []

    accepted: list[Snarl] = []
    append = accepted.append
    extend = accepted.extend
    prefix = sweep_index.tip_prefix
    tip_preorders = sweep_index.tip_preorders
    preorder = sweep_index.preorder
    subtree_end = sweep_index.subtree_end
    point_index = 0
    point_count = len(tip_preorders)

    for interval_index, interval in enumerate(prepared_snarls):
        x = interval.left_boundary
        y = interval.right_boundary
        x_left = interval.left_preorder
        x_right = interval.subtree_end_preorder

        while point_index < point_count and tip_preorders[point_index] < x_left:
            point_index += 1

        if point_index >= point_count:
            extend(
                (remaining.left_boundary, remaining.right_boundary)
                for remaining in prepared_snarls[interval_index:]
            )
            break

        if tip_preorders[point_index] > x_right:
            append((x, y))
            continue

        tips_under_x = prefix[x_right + 1] - prefix[x_left]
        y_left = preorder.get(y)
        if y_left is None:
            continue
        y_right = subtree_end[y]

        if y_left <= x_left and x_right <= y_right:
            append((x, y))
            continue

        if x_left <= y_left and y_right <= x_right:
            tips_under_y = prefix[y_right + 1] - prefix[y_left]
            if tips_under_x == tips_under_y:
                append((x, y))

    return accepted


def _validate_inputs(
    prepared_snarls: list[PreparedSnarlInterval],
    sweep_index: IntervalSweepIndex,
) -> None:
    if prepared_snarls and not isinstance(prepared_snarls[0], PreparedSnarlInterval):
        raise TypeError("prepared_snarls must contain PreparedSnarlInterval objects")
    if not isinstance(sweep_index, IntervalSweepIndex):
        raise TypeError("sweep_index must be an IntervalSweepIndex object")


def _configure_utf8_console() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8")
        except ValueError:
            pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the old pointer sweep with the interval-count sweep improvement."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing .gfa and matching -T.json files.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Dataset base names to run, for example synth1 synth2 synth3.",
    )
    return parser.parse_args()


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else HERE / path


def _run_quietly(func, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _time_call(func, *args, quiet: bool = False, **kwargs):
    start = time.perf_counter()
    result = _run_quietly(func, *args, **kwargs) if quiet else func(*args, **kwargs)
    return result, time.perf_counter() - start


def _compare_dataset(dataset_dir: Path, name: str) -> dict[str, object]:
    gfa_file = dataset_dir / f"{name}.gfa"
    json_file = dataset_dir / f"{name}-T.json"
    if not gfa_file.exists():
        raise FileNotFoundError(f"Missing GFA file: {gfa_file}")
    if not json_file.exists():
        raise FileNotFoundError(f"Missing JSON file: {json_file}")

    graph, graph_s = _time_call(parse_and_build_bi_graph, gfa_file, quiet=True)
    root, root_s = _time_call(find_root, graph, quiet=True)
    if root is None:
        raise RuntimeError(f"No forward root reaches all nodes for {name}")

    _, lca_s = _time_call(precompute_unique_paths_forward, graph, root, quiet=True)
    point_bundle, ftip_s = _time_call(find_leaf_nodes_and_prepared_points, graph, quiet=True)
    ftip, prepared_points = point_bundle
    snarl_bundle, snarls_s = _time_call(
        transform_json_to_snarls_and_prepared,
        json_file,
        graph,
        root,
        mapping_base=None,
        assume_precomputed=True,
        quiet=True,
    )
    raw_snarls, prepared_snarls = snarl_bundle

    old_result, old_s = _time_call(find_ultrabubbles_left_to_right, prepared_snarls, prepared_points)
    interval_index, interval_prepare_s = _time_call(prepare_interval_sweep_index, prepared_points)
    interval_result, interval_s = _time_call(
        find_ultrabubbles_interval_sweep,
        prepared_snarls,
        interval_index,
    )

    old_sorted = sorted(old_result, key=pair_custom_key)
    interval_sorted = sorted(interval_result, key=pair_custom_key)

    return {
        "dataset": name,
        "snarls": len(raw_snarls),
        "rl_snarls": len(prepared_snarls),
        "tips": len(ftip),
        "old_ul": len(old_result),
        "interval_ul": len(interval_result),
        "exact_match": old_sorted == interval_sorted,
        "set_match": set(old_result) == set(interval_result),
        "graph_s": graph_s,
        "root_s": root_s,
        "lca_s": lca_s,
        "ftip_s": ftip_s,
        "snarls_s": snarls_s,
        "interval_prepare_s": interval_prepare_s,
        "old_sweep_s": old_s,
        "interval_sweep_s": interval_s,
    }


def _print_rows(rows: list[dict[str, object]]) -> None:
    headers = [
        "DATASET",
        "SNARLS",
        "R-L",
        "TIPS",
        "OLD_UL",
        "INTERVAL_UL",
        "EXACT",
        "SET",
        "OLD_SWEEP",
        "PREFIX",
        "INTERVAL",
    ]
    table = [
        [
            str(row["dataset"]),
            str(row["snarls"]),
            str(row["rl_snarls"]),
            str(row["tips"]),
            str(row["old_ul"]),
            str(row["interval_ul"]),
            "Y" if row["exact_match"] else "N",
            "Y" if row["set_match"] else "N",
            f"{row['old_sweep_s']:.6f}",
            f"{row['interval_prepare_s']:.6f}",
            f"{row['interval_sweep_s']:.6f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    _configure_utf8_console()
    args = _parse_args()
    dataset_dir = _resolve_path(args.dataset_dir)

    rows = []
    for name in args.names:
        print(f"Running {name} ...")
        rows.append(_compare_dataset(dataset_dir, name))

    print()
    _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
