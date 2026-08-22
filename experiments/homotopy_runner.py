"""
Runner for the Flow Matching x Homotopy Continuation experiment plan
(see HOMOTOPY_EXPERIMENTS.md). Dispatches by experiment number to the
per-experiment script; unrecognized args are forwarded unchanged.

All experiments log to the same MLflow experiment ("homotopy_experiments",
sqlite:///mlflow.db) so results are comparable across runs.

Usage:
    uv run experiments/homotopy_runner.py 1
    uv run experiments/homotopy_runner.py 1 --nfe-budgets 10 20 --beta 5.0
    uv run experiments/homotopy_runner.py 3   # not implemented yet
"""

import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENTS = {
    1: ("adaptive_sampler.py", "Condition-Number-Adaptive Sampler"),
    2: (None, "Predictor-Corrector Sampling"),
    3: (None, "Mode-Matched Base Distribution"),
    4: (None, "Implicit Homotopy Generation (IHG)"),
}


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch one experiment from HOMOTOPY_EXPERIMENTS.md"
    )
    parser.add_argument("experiment", type=int, choices=sorted(EXPERIMENTS))
    args, rest = parser.parse_known_args()

    script, name = EXPERIMENTS[args.experiment]
    if script is None:
        print(f"Experiment {args.experiment} ({name}) is not implemented yet.")
        print("See HOMOTOPY_EXPERIMENTS.md for the plan.")
        sys.exit(1)

    script_path = Path(__file__).parent / script
    print(f"Running experiment {args.experiment}: {name}")
    subprocess.run([sys.executable, str(script_path), *rest], check=True)


if __name__ == "__main__":
    main()
