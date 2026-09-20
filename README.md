# UltraLCA-2

UltraLCA-2 performs ultrabubble detection on a given set of K snarls of a variation graph.
It includes the baseline naive method, sweep-based methods, nested-family methods and hybrid methods.

## Data

This repository includes 13 smaller example datasets inside `GFA_JSON`.
Full dataset (including larger graphs) is hosted on Zenodo: [https://doi.org/10.5281/zenodo.18484366](https://doi.org/10.5281/zenodo.18484366)

Each dataset consists of:
- one `.gfa` variation-graph file in GFA (Graphical Fragment Assembly) format
- one corresponding `-T.json` snarl file

The `-T.json` file contains the snarl information produced by the `vg tool` and used as input by the pipeline.

## Usage

Run the following commands from the root of the repository.

The commands below are shown in PowerShell-style code blocks for readability, but they can also be run from Windows Command Prompt (`cmd`) from the same repository root.

By default, the public runner looks for datasets in `GFA_JSON`.

Available method names:
- `naive`
- `ultralca`
- `sweep`
- `sweep_interval`
- `hybrid_ultralca_online`
- `hybrid_sweep_online`
- `nested_ultralca`
- `nested_hybrid_ultralca`
- `all`

Run one bundled dataset with all methods:

```powershell
python run\run_methods.py --names synth4 --methods all
```

Run one bundled dataset with one method:

```powershell
python run\run_methods.py --names lpa120 --methods ultralca
```

Run one bundled dataset with several methods:

```powershell
python run\run_methods.py --names chr6.C4 --methods ultralca sweep sweep_interval
```

Run several bundled datasets with one or more methods:

```powershell
python run\run_methods.py --names synth1 synth4 lpa120 --methods ultralca sweep sweep_interval
```

Run all bundled datasets in the default `GFA_JSON` folder with all methods:

```powershell
python run\run_methods.py --all --methods all
```

Run all datasets from another folder:

```powershell
python run\run_methods.py --dataset-dir C:\path\to\your\folder --all --methods all
```

Note: the hybrid methods use an `alpha` threshold. If `--alpha` is not provided, the default value `1.7` is used and this is reported in the console.

Example with explicit `alpha`:

```powershell
python run\run_methods.py --names synth4 --methods all --alpha 2.0
```

## Output

For each dataset, the output reports:
- the total number of snarls
- the number of R-L snarls
- the number of ultrabubbles found by each selected method
- the running time of each selected method

When more than one method is selected, the outputs are cross-checked.
If `naive` is included among the selected methods, all other selected methods are checked against `naive`.
If `naive` is not included, the selected methods are cross-checked against each other.

## Benchmarking

Run the preprocessing benchmark:

```powershell
python benchmarks\benchmark_all_preprocesses.py --output-csv benchmarks\benchmark_all_preprocesses_results.csv
```

Run the clean-method benchmark:

```powershell
python benchmarks\benchmark_all_clean_methods.py --alpha 1.7 --output-csv benchmarks\benchmark_all_clean_methods_alpha1p7_results.csv
```

Build the combined benchmark tables after the preprocessing and clean-method benchmarks have already been run. 
This step reads their CSV outputs from the `benchmarks` folder and produces the total benchmark tables by summing preprocessing and clean runtimes:

```powershell
python benchmarks\build_method_tables.py --preprocess-csv benchmarks\benchmark_all_preprocesses_results.csv --clean-csv benchmarks\benchmark_all_clean_methods_alpha1p7_results.csv --output-dir benchmarks --output-prefix benchmark_all_alpha1p7
```

## Requirements

- Python 3.11+
- `networkx` 3.3+

Install if needed:

```powershell
pip install "networkx>=3.3"
```

## Note

The commands above use Windows-style paths.

## Citing UltraLCA-2

### Citation

If you use this code, please cite the paper:

Zisis, A. E., & Sætrom, P. (2026). A rooted tree framework for linear time ultrabubble detection.  
https://arxiv.org/abs/2609.14852

### BibTeX

```bibtex
@misc{zisis2026rootedtreeframework,
      title={A rooted tree framework for linear time ultrabubble detection},
      author={Athanasios E. Zisis and Pål Sætrom},
      year={2026},
      eprint={2609.14852},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      doi={10.48550/arXiv.2609.14852},
      url={https://arxiv.org/abs/2609.14852}
}
```
