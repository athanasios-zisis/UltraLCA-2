from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_common import (
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_DATASET_DIR,
    build_ftip_preorder_points,
    build_rl_preorder_intervals,
    configure_utf8_console,
    discover_dataset_pairs,
    normalize_snarls_rl,
    parse_json_snarls,
    resolve_repo_path,
    time_call,
)
from methods.nested.nestedsnarls_dfs_common import (
    build_dfs_frontier_entry_linear_snarl_forest,
    compute_allowed_dfs_order,
)
from methods.sweep_interval.sweep_interval_prefix import prepare_interval_sweep_index
from pipeline.ftip import find_leaf_nodes
from pipeline.lca import precompute_unique_paths_forward
from pipeline.parse_root import find_root, parse_and_build_bi_graph


DEFAULT_OUTPUT_CSV = DEFAULT_BENCHMARK_DIR / "benchmark_all_preprocesses_results.csv"


@dataclass(frozen=True)
class BenchmarkRow:
    dataset_name: str
    graph_build_s: float
    snarls_parse_s: float
    snarl_rl_normalization_s: float
    root_check_s: float
    rooted_preorder_tables_s: float
    rmq_build_s: float
    ftip_set_s: float
    ftip_preorder_points_s: float
    rl_preorder_intervals_s: float
    prefix_table_s: float
    dfs_order_s: float
    nested_family_build_s: float


@dataclass(frozen=True)
class SkippedDataset:
    dataset_name: str
    reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only the preprocessing subprocesses for every dataset "
            "in GFA_JSON. This script is alpha-independent."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing .gfa and matching -T.json files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV file to write.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional dataset base names to run, for example synth1 synth2 LPA.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately on the first dataset error instead of skipping it.",
    )
    return parser.parse_args()


def _benchmark_dataset(pair) -> BenchmarkRow:
    graph, graph_build_s = time_call(parse_and_build_bi_graph, pair.gfa_file, quiet=True)
    raw_snarls, snarls_parse_s = time_call(parse_json_snarls, pair.json_file)

    root, root_check_s = time_call(find_root, graph, quiet=True)
    if root is None:
        raise RuntimeError(f"No forward root reaches all nodes for {pair.dataset_name}")

    lca_timings = precompute_unique_paths_forward(graph, root, return_timings=True)
    rooted_preorder_tables_s = lca_timings["rooted_preorder_tables_s"]
    rmq_build_s = lca_timings["rmq_build_s"]

    normalization_result, snarl_rl_normalization_s = time_call(normalize_snarls_rl, raw_snarls)
    _canonical_rl_snarls, normalized_raw_snarls = normalization_result

    ftip, ftip_set_s = time_call(find_leaf_nodes, graph)
    prepared_points, ftip_preorder_points_s = time_call(build_ftip_preorder_points, ftip)
    prepared_snarls, rl_preorder_intervals_s = time_call(build_rl_preorder_intervals, normalized_raw_snarls)
    _prefix_index, prefix_table_s = time_call(prepare_interval_sweep_index, prepared_points)
    dfs_result, dfs_order_s = time_call(compute_allowed_dfs_order, graph, root)
    _nested_bundle, nested_family_build_s = time_call(
        build_dfs_frontier_entry_linear_snarl_forest,
        prepared_snarls,
        dfs_result,
    )

    return BenchmarkRow(
        dataset_name=pair.dataset_name,
        graph_build_s=graph_build_s,
        snarls_parse_s=snarls_parse_s,
        snarl_rl_normalization_s=snarl_rl_normalization_s,
        root_check_s=root_check_s,
        rooted_preorder_tables_s=rooted_preorder_tables_s,
        rmq_build_s=rmq_build_s,
        ftip_set_s=ftip_set_s,
        ftip_preorder_points_s=ftip_preorder_points_s,
        rl_preorder_intervals_s=rl_preorder_intervals_s,
        prefix_table_s=prefix_table_s,
        dfs_order_s=dfs_order_s,
        nested_family_build_s=nested_family_build_s,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _print_console_table(rows: list[BenchmarkRow]) -> None:
    headers = [
        "DATASET",
        "GRAPH",
        "PARSE",
        "R-L",
        "ROOT",
        "RPRE",
        "RMQ",
        "FTIP",
        "PTS",
        "INTV",
        "PFX",
        "DFS",
        "NESTED",
    ]
    table = [
        [
            row.dataset_name,
            _format_seconds(row.graph_build_s),
            _format_seconds(row.snarls_parse_s),
            _format_seconds(row.snarl_rl_normalization_s),
            _format_seconds(row.root_check_s),
            _format_seconds(row.rooted_preorder_tables_s),
            _format_seconds(row.rmq_build_s),
            _format_seconds(row.ftip_set_s),
            _format_seconds(row.ftip_preorder_points_s),
            _format_seconds(row.rl_preorder_intervals_s),
            _format_seconds(row.prefix_table_s),
            _format_seconds(row.dfs_order_s),
            _format_seconds(row.nested_family_build_s),
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table)) if table else len(header)
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in table:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _write_csv(rows: list[BenchmarkRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    configure_utf8_console()
    args = _parse_args()

    dataset_dir = resolve_repo_path(args.dataset_dir)
    output_csv = resolve_repo_path(args.output_csv)

    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    pairs = discover_dataset_pairs(dataset_dir, args.names)
    if not pairs:
        print(f"No dataset pairs found in {dataset_dir}")
        return 1

    rows: list[BenchmarkRow] = []
    skipped: list[SkippedDataset] = []
    total = len(pairs)

    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{total}] Running {pair.dataset_name} ...")
        try:
            row = _benchmark_dataset(pair)
        except Exception as exc:
            if args.stop_on_error:
                raise
            reason = str(exc).strip() or exc.__class__.__name__
            skipped.append(SkippedDataset(pair.dataset_name, reason))
            print(f"    SKIPPED: {reason}")
            continue

        rows.append(row)
        print(
            f"    parse={row.snarls_parse_s:.6f}s "
            f"rooted_preorder={row.rooted_preorder_tables_s:.6f}s "
            f"rmq={row.rmq_build_s:.6f}s"
        )

    if not rows:
        print("No successful datasets to write.")
        if skipped:
            print("Skipped datasets:")
            for item in skipped:
                print(f"  - {item.dataset_name}: {item.reason}")
        return 1

    print()
    _print_console_table(rows)
    _write_csv(rows, output_csv)
    print()
    print(f"CSV written to: {output_csv}")
    if skipped:
        print()
        print("Skipped datasets:")
        for item in skipped:
            print(f"  - {item.dataset_name}: {item.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
