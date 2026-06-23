from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.benchmark_common import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_DATASET_DIR,
    SharedArtifacts,
    build_shared_artifacts,
    configure_utf8_console,
    discover_dataset_pairs,
    resolve_repo_path,
    time_call,
)
from methods.hybrid.ultrabubbles_hybrid_sweep_online import (  # noqa: E402
    find_ultrabubbles_bounded_hybrid_naive_sweep_online,
)
from methods.hybrid.ultrabubbles_hybrid_ultralca_online import (  # noqa: E402
    find_ultrabubbles_bounded_hybrid_naive_ultralca_online,
)
from methods.naive.ultrabubbles_naive_dag import find_ultra  # noqa: E402
from methods.nested.ultrabubbles_nested_hybrid_ultralca import (  # noqa: E402
    find_ultrabubbles_nested_bounded_hybrid_naive_ultralca,
)
from methods.nested.ultrabubbles_nested_ultralca import (  # noqa: E402
    find_ultrabubbles_nested_ultralca,
)
from methods.sweep.ultrabubbles_sweep import find_ultrabubbles_left_to_right  # noqa: E402
from methods.sweep_interval.sweep_interval import find_ultrabubbles_interval_sweep  # noqa: E402
from methods.ultralca.ultrabubbles_ultralca import find_ultrabubbles_ultralca  # noqa: E402
from pipeline.common import pair_custom_key  # noqa: E402


Snarl = tuple[str, str]

ALL_METHODS = [
    "naive",
    "ultralca",
    "sweep",
    "sweep_interval",
    "hybrid_ultralca_online",
    "hybrid_sweep_online",
    "nested_ultralca",
    "nested_hybrid_ultralca",
]
HYBRID_METHODS = {
    "hybrid_ultralca_online",
    "hybrid_sweep_online",
    "nested_hybrid_ultralca",
}
METHOD_LABELS = {
    "naive": "naive",
    "ultralca": "UltraLCA",
    "sweep": "SWEEP",
    "sweep_interval": "SWEEP_INTERVAL",
    "hybrid_ultralca_online": "HYBRID_UltraLCA_ONLINE",
    "hybrid_sweep_online": "HYBRID_SWEEP_ONLINE",
    "nested_ultralca": "NESTED_UltraLCA",
    "nested_hybrid_ultralca": "NESTED_HYBRID_UltraLCA",
}


@dataclass(frozen=True)
class MethodRunResult:
    method: str
    ultrabubbles: list[Snarl]
    runtime_s: float


@dataclass(frozen=True)
class DatasetRunResult:
    dataset_name: str
    total_snarls: int
    rl_snarls: int
    ultrabubble_runs: list[MethodRunResult]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more ultrabubble methods on one dataset, many named datasets, "
            "or all datasets in a directory."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing .gfa files and their matching -T.json files.",
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--names",
        nargs="+",
        help="Dataset base names to run, for example synth1 synth2 LPA.",
    )
    dataset_group.add_argument(
        "--all",
        action="store_true",
        help="Run every dataset pair found in the dataset directory.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help=(
            "Methods to run. Use one or more of: "
            "naive ultralca sweep sweep_interval "
            "hybrid_ultralca_online hybrid_sweep_online "
            "nested_ultralca nested_hybrid_ultralca. "
            "You can also pass all."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Hybrid alpha parameter. Used only by hybrid methods. Defaults to 1.7 when omitted.",
    )
    parser.add_argument(
        "--show-ultrabubbles",
        action="store_true",
        help="Print the ultrabubble pairs for each dataset and method.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately on the first dataset error instead of skipping it.",
    )
    return parser


def _parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = _build_parser()
    return parser, parser.parse_args()


def _normalize_method_name(token: str) -> str | None:
    normalized = token.strip().casefold().replace("-", "_")
    if normalized in {"all", "*"}:
        return "all"
    for method in ALL_METHODS:
        if normalized == method.casefold():
            return method
    return None


def _resolve_methods(tokens: list[str], parser: argparse.ArgumentParser) -> list[str]:
    if not tokens:
        parser.error("At least one method must be provided with --methods.")

    resolved: list[str] = []
    seen: set[str] = set()
    saw_all = False

    for token in tokens:
        method = _normalize_method_name(token)
        if method is None:
            parser.error(f"Unknown method: {token}")
        if method == "all":
            saw_all = True
            continue
        if method not in seen:
            resolved.append(method)
            seen.add(method)

    if saw_all:
        return ALL_METHODS[:]
    return resolved


def _normalize_output(result: list[Snarl]) -> list[Snarl]:
    return sorted(set(result), key=pair_custom_key)


def _run_naive(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultra(artifacts.graph, artifacts.normalized_raw_snarls, True)


def _run_ultralca(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_ultralca(
        artifacts.graph,
        artifacts.normalized_raw_snarls,
        artifacts.ftip,
    )


def _run_sweep(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_left_to_right(
        artifacts.prepared_snarls,
        artifacts.prepared_points,
    )


def _run_sweep_interval(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_interval_sweep(
        artifacts.prepared_snarls,
        artifacts.prefix_index,
    )


def _run_hybrid_ultralca(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_bounded_hybrid_naive_ultralca_online(
        artifacts.graph,
        artifacts.normalized_raw_snarls,
        artifacts.ftip,
        artifacts.prepared_snarls,
        alpha=alpha,
        tips_count=len(artifacts.ftip),
    )


def _run_hybrid_sweep(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_bounded_hybrid_naive_sweep_online(
        artifacts.graph,
        artifacts.prepared_snarls,
        artifacts.prepared_points,
        alpha=alpha,
    )


def _run_nested_ultralca(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_nested_ultralca(
        artifacts.nested_nodes,
        artifacts.nested_families,
        artifacts.ftip,
    )


def _run_nested_hybrid_ultralca(artifacts: SharedArtifacts, alpha: float) -> list[Snarl]:
    return find_ultrabubbles_nested_bounded_hybrid_naive_ultralca(
        artifacts.graph,
        artifacts.nested_nodes,
        artifacts.nested_families,
        artifacts.ftip,
        alpha=alpha,
        tips_count=len(artifacts.ftip),
    )


METHOD_RUNNERS: dict[str, Callable[[SharedArtifacts, float], list[Snarl]]] = {
    "naive": _run_naive,
    "ultralca": _run_ultralca,
    "sweep": _run_sweep,
    "sweep_interval": _run_sweep_interval,
    "hybrid_ultralca_online": _run_hybrid_ultralca,
    "hybrid_sweep_online": _run_hybrid_sweep,
    "nested_ultralca": _run_nested_ultralca,
    "nested_hybrid_ultralca": _run_nested_hybrid_ultralca,
}


def _print_dataset_result(result: DatasetRunResult, show_ultrabubbles: bool) -> None:
    print()
    print(f"Dataset: {result.dataset_name}")
    print(f"  Total snarls: {result.total_snarls}")
    print(f"  R-L snarls:   {result.rl_snarls}")
    print()

    headers = ["METHOD", "ULTRABUBBLES", "RUNTIME(s)"]
    rows = [
        [
            METHOD_LABELS[item.method],
            str(len(item.ultrabubbles)),
            f"{item.runtime_s:.6f}",
        ]
        for item in result.ultrabubble_runs
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print("  " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  " + "  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    if not show_ultrabubbles:
        return

    for item in result.ultrabubble_runs:
        print()
        print(f"  {METHOD_LABELS[item.method]} ultrabubbles:")
        if not item.ultrabubbles:
            print("    (none)")
            continue
        for left_boundary, right_boundary in item.ultrabubbles:
            print(f"    {left_boundary}  {right_boundary}")


def _run_dataset(pair, methods: list[str], alpha: float) -> DatasetRunResult:
    artifacts = build_shared_artifacts(pair)
    runs: list[MethodRunResult] = []

    for method in methods:
        result, runtime_s = time_call(METHOD_RUNNERS[method], artifacts, alpha)
        runs.append(
            MethodRunResult(
                method=method,
                ultrabubbles=_normalize_output(result),
                runtime_s=runtime_s,
            )
        )

    return DatasetRunResult(
        dataset_name=pair.dataset_name,
        total_snarls=len(artifacts.raw_snarls),
        rl_snarls=len(artifacts.prepared_snarls),
        ultrabubble_runs=runs,
    )


def main() -> int:
    configure_utf8_console()
    parser, args = _parse_args()

    methods = _resolve_methods(args.methods, parser)
    dataset_dir = resolve_repo_path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return 1

    uses_hybrid = any(method in HYBRID_METHODS for method in methods)
    alpha = args.alpha if args.alpha is not None else DEFAULT_ALPHA
    if uses_hybrid and args.alpha is None:
        print(f"No --alpha provided; using default alpha={DEFAULT_ALPHA}.")

    selected_names = None if args.all else args.names
    pairs = discover_dataset_pairs(dataset_dir, selected_names)
    if not pairs:
        print(f"No dataset pairs found in {dataset_dir}")
        return 1

    total = len(pairs)
    successful_runs: list[DatasetRunResult] = []

    for index, pair in enumerate(pairs, start=1):
        print(f"[{index}/{total}] Running {pair.dataset_name} ...")
        try:
            dataset_result = _run_dataset(pair, methods, alpha)
        except Exception as exc:
            if args.stop_on_error:
                raise
            reason = str(exc).strip() or exc.__class__.__name__
            print(f"    SKIPPED: {reason}")
            continue
        successful_runs.append(dataset_result)
        counts_summary = ", ".join(
            f"{METHOD_LABELS[item.method]}={len(item.ultrabubbles)}"
            for item in dataset_result.ultrabubble_runs
        )
        print(f"    {counts_summary}")

    if not successful_runs:
        print("No successful datasets to report.")
        return 1

    for dataset_result in successful_runs:
        _print_dataset_result(dataset_result, args.show_ultrabubbles)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
