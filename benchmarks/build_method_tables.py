from __future__ import annotations

import argparse
import csv
from pathlib import Path

from benchmark_common import (
    DEFAULT_ALPHA,
    DEFAULT_BENCHMARK_DIR,
    NON_NAIVE_METHOD_STEMS,
    alpha_tag,
    configure_utf8_console,
    resolve_repo_path,
)


PREPROCESS_COLUMNS_BY_METHOD = {
    "ultralca": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
    ],
    "sweep": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
        "ftip_preorder_points_s",
        "rl_preorder_intervals_s",
    ],
    "sweep_interval": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "ftip_set_s",
        "ftip_preorder_points_s",
        "rl_preorder_intervals_s",
        "prefix_table_s",
    ],
    "hybrid_ultralca_online": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
    ],
    "hybrid_sweep_online": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
        "ftip_preorder_points_s",
        "rl_preorder_intervals_s",
    ],
    "nested_ultralca": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
        "dfs_order_s",
        "nested_family_build_s",
    ],
    "nested_hybrid_ultralca": [
        "graph_build_s",
        "snarls_parse_s",
        "snarl_rl_normalization_s",
        "root_check_s",
        "rooted_preorder_tables_s",
        "rmq_build_s",
        "ftip_set_s",
        "dfs_order_s",
        "nested_family_build_s",
    ],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build method-level preprocessing and total-time CSV tables from the "
            "preprocessing and clean benchmark outputs. No graph reruns are performed."
        )
    )
    parser.add_argument(
        "--preprocess-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / "benchmark_all_preprocesses_results.csv",
        help="CSV produced by benchmark_all_preprocesses.py.",
    )
    parser.add_argument(
        "--clean-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR / f"benchmark_all_clean_methods_alpha{alpha_tag(DEFAULT_ALPHA)}_results.csv",
        help="CSV produced by benchmark_all_clean_methods.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BENCHMARK_DIR,
        help="Directory for the derived CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional prefix for the derived CSV filenames.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional dataset base names to keep in the derived outputs.",
    )
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bool_from_csv(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _resolve_output_prefix(args: argparse.Namespace, alpha_values: set[str]) -> str:
    if args.output_prefix:
        return args.output_prefix
    if len(alpha_values) == 1:
        alpha = next(iter(alpha_values))
        return f"benchmark_method_alpha{alpha_tag(float(alpha))}"
    return "benchmark_method"


def main() -> int:
    configure_utf8_console()
    args = _parse_args()

    preprocess_csv = resolve_repo_path(args.preprocess_csv)
    clean_csv = resolve_repo_path(args.clean_csv)
    output_dir = resolve_repo_path(args.output_dir)

    if not preprocess_csv.exists():
        print(f"Missing preprocessing CSV: {preprocess_csv}")
        return 1
    if not clean_csv.exists():
        print(f"Missing clean benchmark CSV: {clean_csv}")
        return 1

    preprocess_rows = _read_csv_rows(preprocess_csv)
    clean_rows = _read_csv_rows(clean_csv)
    selected_names = {name.casefold() for name in args.names} if args.names else None

    preprocess_by_name = {
        row["dataset_name"]: row
        for row in preprocess_rows
        if selected_names is None or row["dataset_name"].casefold() in selected_names
    }
    clean_by_name = {
        row["dataset_name"]: row
        for row in clean_rows
        if selected_names is None or row["dataset_name"].casefold() in selected_names
    }

    dataset_names = sorted(set(preprocess_by_name) & set(clean_by_name), key=str.casefold)
    if not dataset_names:
        print("No overlapping datasets between the preprocessing and clean benchmark CSVs.")
        return 1

    alpha_values = {clean_by_name[name]["alpha"] for name in dataset_names}
    output_prefix = _resolve_output_prefix(args, alpha_values)

    preprocessing_rows_out: list[dict[str, object]] = []
    total_rows_out: list[dict[str, object]] = []
    combined_rows_out: list[dict[str, object]] = []

    for dataset_name in dataset_names:
        preprocess_row = preprocess_by_name[dataset_name]
        clean_row = clean_by_name[dataset_name]
        alpha = float(clean_row["alpha"])

        preprocessing_out = {"dataset_name": dataset_name}
        total_out = {
            "dataset_name": dataset_name,
            "alpha": alpha,
        }

        for method in NON_NAIVE_METHOD_STEMS:
            preprocessing_total = sum(float(preprocess_row[column]) for column in PREPROCESS_COLUMNS_BY_METHOD[method])
            clean_time = float(clean_row[f"{method}_clean_s"])
            total_time = preprocessing_total + clean_time
            ul_count = int(clean_row[f"{method}_ul_count"])
            matches_naive = _bool_from_csv(clean_row[f"{method}_matches_naive"])

            preprocessing_out[f"{method}_pre_s"] = preprocessing_total
            total_out[f"{method}_total_s"] = total_time

            combined_rows_out.append(
                {
                    "dataset_name": dataset_name,
                    "alpha": alpha,
                    "method": method,
                    "pre_s": preprocessing_total,
                    "clean_s": clean_time,
                    "total_s": total_time,
                    "ul_count": ul_count,
                    "matches_naive": matches_naive,
                    "total_snarls": int(clean_row["total_snarls"]),
                    "rl_snarls": int(clean_row["rl_snarls"]),
                }
            )

        total_out["total_snarls"] = int(clean_row["total_snarls"])
        total_out["rl_snarls"] = int(clean_row["rl_snarls"])
        total_out["ultrabubbles"] = int(clean_row["naive_ul_count"])

        preprocessing_rows_out.append(preprocessing_out)
        total_rows_out.append(total_out)

    preprocess_output_csv = output_dir / f"{output_prefix}_preprocessing_totals.csv"
    total_output_csv = output_dir / f"{output_prefix}_total_times.csv"
    combined_output_csv = output_dir / f"{output_prefix}_combined_long.csv"

    _write_rows(
        preprocess_output_csv,
        ["dataset_name"] + [f"{method}_pre_s" for method in NON_NAIVE_METHOD_STEMS],
        preprocessing_rows_out,
    )
    _write_rows(
        total_output_csv,
        ["dataset_name", "alpha"]
        + [f"{method}_total_s" for method in NON_NAIVE_METHOD_STEMS]
        + ["total_snarls", "rl_snarls", "ultrabubbles"],
        total_rows_out,
    )
    _write_rows(
        combined_output_csv,
        [
            "dataset_name",
            "alpha",
            "method",
            "pre_s",
            "clean_s",
            "total_s",
            "ul_count",
            "matches_naive",
            "total_snarls",
            "rl_snarls",
        ],
        combined_rows_out,
    )

    print(f"Preprocessing totals CSV written to: {preprocess_output_csv}")
    print(f"Total-time CSV written to: {total_output_csv}")
    print(f"Combined long-format CSV written to: {combined_output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
