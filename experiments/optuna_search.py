"""
Hyperparameter search with Optuna. The objective is the FID of the final weights.

Each trial trains with scripts/train.py, samples with scripts/fid.py and scores with PyTorch,
each in its own process, one after another. The learning rate decays to 0 at the end of every
run (lr_scheduler=warmup_cosine), so one score per trial is reliable to about 1 FID.

The study is stored in SQLite: run the same command again to add trials to it. A trial that
was stopped is tried again once.

Usage:
    BEAM=2 uv run --group search experiments/optuna_search.py --study lr --trials 8
    BEAM=2 uv run --group search experiments/optuna_search.py --study recipe \\
        --params lr grad_clip dropout hflip path --trials 20
    uv run --group search experiments/optuna_search.py --study lr --report
"""

import argparse
import json
import os
import subprocess
import sys
import warnings

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback

# The heartbeat that retries stopped trials is marked experimental
warnings.filterwarnings("ignore", category=ExperimentalWarning)

OUT_DIR = os.path.join("outputs", "optuna")
STORAGE = f"sqlite:///{OUT_DIR}/optuna.db"


def suggest(trial: optuna.Trial, params: list[str], lr_range=(3e-4, 3e-3)) -> list[str]:
    """Hydra overrides for the parameters under search."""
    overrides = []
    if "lr" in params:
        overrides.append(f"optimizer.lr={trial.suggest_float('lr', *lr_range, log=True)}")
    if "grad_clip" in params:
        clip = trial.suggest_categorical("grad_clip", [None, 1.0])
        overrides.append(f"training.grad_clip={'null' if clip is None else clip}")
    if "dropout" in params:
        overrides.append(f"model.dropout={trial.suggest_categorical('dropout', [0.0, 0.1])}")
    if "hflip" in params:
        hflip = trial.suggest_categorical("hflip", [False, True])
        overrides.append(f"dataset.hflip={str(hflip).lower()}")
    if "path" in params:
        overrides.append(f"scheduler={trial.suggest_categorical('path', ['linear', 'lvp'])}")
    return overrides


def run(cmd: list[str]):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def objective(trial: optuna.Trial, args) -> float:
    name = f"{args.study}_t{trial.number:03d}"
    weights = os.path.join("weights", f"optuna_{name}.safetensors")
    samples = os.path.join(OUT_DIR, args.study, f"{name}.npz")
    train = [
        "dataset=cifar10",
        "model=unet",
        "training=cifar10_default",
        f"training.num_epochs={args.epochs}",
        "lr_scheduler=warmup_cosine",
        "training.ema_decay=null",
        f"seed={args.seed}",
        f"mlflow.experiment_name=optuna_{args.study}",
        f"+model_name=optuna_{name}.safetensors",
        f"output_dir={os.path.join(OUT_DIR, args.study, name)}",
        *args.set,
        *suggest(trial, args.params, args.lr_range),
    ]
    run([sys.executable, "scripts/train.py", *train])

    sample = ["scripts/fid.py", "sample", "--weights", weights, "--out", samples]
    sample += ["--n", str(args.fid_n)]
    if args.model_config:
        sample += ["--model-config", args.model_config]
    run([sys.executable, *sample])
    run(["uv", "run", "--group", "eval", "python", "scripts/fid.py", "score", samples])

    with open(os.path.splitext(samples)[0] + "_scores.json") as f:
        scores = json.load(f)
    os.remove(samples)
    trial.set_user_attr("kid_x1000", scores["kid_x1000"])
    trial.set_user_attr("weights", weights)
    return scores["fid"]


def report(study: optuna.Study):
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    for t in sorted(trials, key=lambda t: t.value or float("inf")):
        params = " ".join(
            f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in t.params.items()
        )
        print(f"#{t.number:3d}  FID {t.value:6.2f}  KID {t.user_attrs['kid_x1000']:6.2f}  {params}")
    failed = len(study.trials) - len(trials)
    if failed:
        print(f"{failed} trials not complete (failed, running or stopped)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--study", required=True, help="study name; trials are added to it")
    parser.add_argument(
        "--params",
        nargs="+",
        default=["lr"],
        choices=["lr", "grad_clip", "dropout", "hflip", "path"],
    )
    parser.add_argument(
        "--lr-range",
        nargs=2,
        type=float,
        default=[3e-4, 3e-3],
        metavar=("LOW", "HIGH"),
        help="log-uniform lr range; use a new --study name when you change it",
    )
    parser.add_argument("--trials", type=int, default=8, help="trials to run in this call")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--seed", type=int, default=0, help="training seed, the same in every trial"
    )
    parser.add_argument("--fid-n", type=int, default=10_000, help="samples for the FID")
    parser.add_argument(
        "--model-config", help="configs/model/<name>.yaml when --set changes the model"
    )
    parser.add_argument(
        "--set", nargs="*", default=[], help="fixed Hydra overrides for every trial"
    )
    parser.add_argument("--report", action="store_true", help="print the completed trials and stop")
    args = parser.parse_args()

    os.makedirs(os.path.join(OUT_DIR, args.study), exist_ok=True)
    storage = RDBStorage(
        STORAGE,
        heartbeat_interval=60,
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(max_retry=1),
    )
    study = optuna.create_study(
        study_name=args.study,
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),
        load_if_exists=True,
    )
    if not args.report:
        study.optimize(lambda trial: objective(trial, args), n_trials=args.trials)
    report(study)


if __name__ == "__main__":
    main()
