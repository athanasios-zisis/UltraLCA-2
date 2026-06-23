from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark_common import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK_DIR,
    DEFAULT_DATASET_DIR,
    alpha_tag,
    build_shared_artifacts,
    configure_utf8_console,
    discover_dataset_pairs,
    resolve_repo_path,
    time_call,
)
from methods.hybrid.ultrabubbles_hybrid_sweep_online import find_ultrabubbles_bounded_hybrid_naive_sweep_online
from methods.hybrid.ultrabubbles_hybrid_ultralca_online import find_ultrabubbles_bounded_hybrid_naive_ultralca_online
from methods.naive.ultrabubbles_naive_dag import find_ultra
from methods.nested.ultrabubbles_nested_hybrid_ultralca import find_ultrabubbles_nested_bounded_hybrid_naive_ultralca
from methods.nested.ultrabubbles_nested_ultralca import find_ultrabubbles_nested_ultralca
from methods.sweep.ultrabubbles_sweep import find_ultrabubbles_left_to_right
from methods.sweep_interval.sweep_interval import find_ultrabubbles_interval_sweep
from methods.ultralca.ultrabubbles_ultralca import find_ultrabubbles_ultralca
from pipeline.common import pair_custom_key


@dataclass(frozen=True)
class CleanBenchmarkRow:
    dataset_name: str
    alpha: float
    naive_clean_s: float
    ultralca_clean_s: float
    sweep_clean_s: float
    sweep_interval_clean_s: float
    hybrid_ultralca_online_clean_s: float
    hybrid_sweep_online_clean_s: float
    nested_ultralca_clean_s: float
    nested_hybrid_ultralca_clean_s: float
    total_snarls: int
    rl_snarls: int
    naive_ul_count: int
    ultralca_ul_count: int
    sweep_ul_count: int
    sweep_interval_ul_count: int
    hybrid_ultralca_online_ul_count: int
    hybrid_sweep_online_ul_count: int
    nested_ultralca_ul_count: int
    nested_hybrid_ultralca_ul_count: int
    naive_matches_naive: bool
    ultralca_matches_naive: bool
    sweep_matches_naive: bool
    sweep_interval_matches_naive: bool
    hybrid_ultralca_online_matches_naive: bool
    hybrid_sweep_online_matches_naive: bool
    nested_ultralca_matches_naive: bool
    nested_hybrid_ultralca_matches_naive: bool


@dataclass(frozen=True)
class SkippedDataset:
    dataset_name: str
    reason: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark only the clean runtime of every method. "
            "Hybrid methods accept --alpha; when omitted the default alpha=1.7 is used."
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
        default=None,
        help="CSV file to write. Default includes the alpha value in the filename.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional dataset base names to run, for example synth1 synth2 LPA.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Hybrid alpha parameter. If omitted, default alpha=1.7 is used.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately on the first dataset error instead of skipping it.",
    )
    return parser.parse_args()


def _normalize_output(result: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(set(result), key=pair_custom_key)


def _benchmark_dataset(pair, alpha: float) -> CleanBenchmarkRow:
    artifacts = build_shared_artifacts(pair)
    total_snarls = len(artifacts.raw_snarls)
    rl_snarls = len(artifacts.prepared_snarls)
    tips_count = len(artifacts.ftip)

    naive_result, naive_clean_s = time_call(find_ultra, artifacts.graph, artifacts.normalized_raw_snarls, True)
    ultralca_result, ultralca_clean_s = time_call(
        find_ultrabubbles_ultralca,
        artifacts.graph,
        artifacts.normalized_raw_snarls,
        artifacts.ftip,
    )
    sweep_result, sweep_clean_s = time_call(
        find_ultrabubbles_left_to_right,
        artifacts.prepared_snarls,
        artifacts.prepared_points,
    )
    sweep_interval_result, sweep_interval_clean_s = time_call(
        find_ultrabubbles_interval_sweep,
        artifacts.prepared_snarls,
        artifacts.prefix_index,
    )
    hybrid_ultralca_result, hybrid_ultralca_clean_s = time_call(
        find_ultrabubbles_bounded_hybrid_naive_ultralca_online,
        artifacts.graph,
        artifacts.normalized_raw_snarls,
        artifacts.ftip,
        artifacts.prepared_snarls,
        alpha=alpha,
        tips_count=tips_count,
    )
    hybrid_sweep_result, hybrid_sweep_clean_s = time_call(
        find_ultrabubbles_bounded_hybrid_naive_sweep_online,
        artifacts.graph,
        artifacts.prepared_snarls,
        artifacts.prepared_points,
        alpha=alpha,
    )
    nested_ultralca_result, nested_ultralca_clean_s = time_call(
        find_ultrabubbles_nested_ultralca,
        artifacts.nested_nodes,
        artifacts.nested_families,
        artifacts.ftip,
    )
    nested_hybrid_ultralca_result, nested_hybrid_ultralca_clean_s = time_call(
        find_ultrabubbles_nested_bounded_hybrid_naive_ultralca,
        artifacts.graph,
        artifacts.nested_nodes,
        artifacts.nested_families,
        artifacts.ftip,
        alpha=alpha,
        tips_count=tips_count,
    )

    normalized_results = {
        "naive": _normalize_output(naive_result),
        "ultralca": _normalize_output(ultralca_result),
        "sweep": _normalize_output(sweep_result),
        "sweep_interval": _normalize_output(sweep_interval_result),
        "hybrid_ultralca_online": _normalize_output(hybrid_ultralca_result),
        "hybrid_sweep_online": _normalize_output(hybrid_sweep_result),
        "nested_ultralca": _normalize_output(nested_ultralca_result),
        "nested_hybrid_ultralca": _normalize_output(nested_hybrid_ultralca_result),
    }
    naive_set = set(normalized_results["naive"])

    return CleanBenchmarkRow(
        dataset_name=pair.dataset_name,
        alpha=alpha,
        naive_clean_s=naive_clean_s,
        ultralca_clean_s=ultralca_clean_s,
        sweep_clean_s=sweep_clean_s,
        sweep_interval_clean_s=sweep_interval_clean_s,
        hybrid_ultralca_online_clean_s=hybrid_ultralca_clean_s,
        hybrid_sweep_online_clean_s=hybrid_sweep_clean_s,
        nested_ultralca_clean_s=nested_ultralca_clean_s,
        nested_hybrid_ultralca_clean_s=nested_hybrid_ultralca_clean_s,
        total_snarls=total_snarls,
        rl_snarls=rl_snarls,
        naive_ul_count=len(normalized_results["naive"]),
        ultralca_ul_count=len(normalized_results["ultralca"]),
        sweep_ul_count=len(normalized_results["sweep"]),
        sweep_interval_ul_count=len(normalized_results["sweep_interval"]),
        hybrid_ultralca_online_ul_count=len(normalized_results["hybrid_ultralca_online"]),
        hybrid_sweep_online_ul_count=len(normalized_results["hybrid_sweep_online"]),
        nested_ultralca_ul_count=len(normalized_results["nested_ultralca"]),
        nested_hybrid_ultralca_ul_count=len(normalized_results["nested_hybrid_ultralca"]),
        naive_matches_naive=True,
        ultralca_matches_naive=set(normalized_results["ultralca"]) == naive_set,
        sweep_matches_naive=set(normalized_results["sweep"]) == naive_set,
        sweep_interval_matches_naive=set(normalized_results["sweep_interval"]) == naive_set,
        hybrid_ultralca_online_matches_naive=set(normalized_results["hybrid_ultralca_online"]) == naive_set,
        hybrid_sweep_online_matches_naive=set(normalized_results["hybrid_sweep_online"]) == naive_set,
        nested_ultralca_matches_naive=set(normalized_results["nested_ultralca"]) == naive_set,
        nested_hybrid_ultralca_matches_naive=set(normalized_results["nested_hybrid_ultralca"]) == naive_set,
    )


def _format_seconds(value: float) -> str:
    return f"{value:.6f}"


def _print_console_table(rows: list[CleanBenchmarkRow]) -> None:
    headers = [
        "DATASET",
        "NAIVE",
        "ULTRALCA",
        "SWEEP",
        "S_INT",
        "H_ULCA",
        "H_SWEEP",
        "N_ULCA",
        "N_HYB",
    ]
    table = [
        [
            row.dataset_name,
            _format_seconds(row.naive_clean_s),
            _format_seconds(row.ultralca_clean_s),
            _format_seconds(row.sweep_clean_s),
            _format_seconds(row.sweep_interval_clean_s),
            _format_seconds(row.hybrid_ultralca_online_clean_s),
            _format_seconds(row.hybrid_sweep_online_clean_s),
            _format_seconds(row.nested_ultralca_clean_s),
            _format_seconds(row.nested_hybrid_ultralca_clean_s),
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


def _write_csv(rows: list[CleanBenchmarkRow], output_csv: Path) -> None:
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
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    alpha = args.alpha
    if alpha is None:
        alpha = DEFAULT_ALPHA
        print(f"No --alpha provided; using default alpha={DEFAULT_ALPHA}.")

    output_csv = (
        resolve_repo_path(args.output_csv)
        if args.output_csv is not None
        else DEFAULT_BENCHMARK_DIR / f"benchmark_all_clean_methods_alpha{alpha_tag(alpha)}_results.csv"
    )

    pairs = discover_dataset_pairs(dataset_dir, args.names)
    if not pairs:
        print(f"No dataset pairs found in {dataset_dir}")
        return 1

    rows: list[CleanBenchmarkRow] = []
    skipped: list[SkippedDataset] = []
    total = len(pairs)

    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{total}] Running {pair.dataset_name} ...")
        try:
            row = _benchmark_dataset(pair, alpha)
        except Exception as exc:
            if args.stop_on_error:
                raise
            reason = str(exc).strip() or exc.__class__.__name__
            skipped.append(SkippedDataset(pair.dataset_name, reason))
            print(f"    SKIPPED: {reason}")
            continue

        rows.append(row)
        print(
            f"    naive={row.naive_clean_s:.6f}s "
            f"sweep_interval={row.sweep_interval_clean_s:.6f}s "
            f"match={row.nested_hybrid_ultralca_matches_naive}"
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
