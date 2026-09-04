"""Kaggle runner for the V2 TCN-LTR custom 5-fold experiment.

Before changing RUN_MODE, add a private Dataset containing
`tcn_ltr_pretrained_tvsum_summe.pt` to kernel-metadata.json. This runner refuses
to fine-tune from random weights so the reported experiment follows the locked
transfer-learning protocol.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working/v2_project")
RUN_MODE = "validate"  # validate | fold | five_fold
FOLD_INDEX = 0


def run(*args: str) -> None:
    subprocess.run(args, cwd=WORKING, check=True)


def find_pretrained_checkpoint() -> Path:
    matches = list(INPUT_ROOT.rglob("tcn_ltr_pretrained_tvsum_summe.pt"))
    if len(matches) != 1:
        raise FileNotFoundError("attach exactly one Dataset containing tcn_ltr_pretrained_tvsum_summe.pt")
    return matches[0]


def run_fold(fold: int, pretrained: Path) -> None:
    manifest = f"data/manifests/custom_fold{fold}.jsonl"
    checkpoint = f"data/models/tcn_ltr_custom_fold{fold}.pt"
    run(
        "python",
        "-m",
        "highlight_agent.models.train_tcn_ltr",
        "--manifest",
        manifest,
        "--cache-dir",
        "data/features_cache",
        "--init-checkpoint",
        str(pretrained),
        "--output",
        checkpoint,
        "--last-output",
        f"data/models/tcn_ltr_custom_fold{fold}_last.pt",
        "--training-log",
        f"data/reports/tcn_ltr_custom_fold{fold}_log.json",
        "--history-csv",
        f"data/reports/tcn_ltr_custom_fold{fold}_history.csv",
        "--training-plot",
        f"data/reports/tcn_ltr_custom_fold{fold}_curves.svg",
        "--max-epochs",
        "50",
        "--patience",
        "15",
        "--lr",
        "1e-4",
    )
    run(
        "python",
        "-m",
        "evaluation.evaluate_tcn_ltr",
        "--manifest",
        manifest,
        "--cache-dir",
        "data/features_cache",
        "--checkpoint",
        checkpoint,
        "--split",
        "test",
        "--output-json",
        f"data/reports/tcn_ltr_custom_fold{fold}_test.json",
    )


if WORKING.exists():
    shutil.rmtree(WORKING)
WORKING.mkdir(parents=True)
manifest_matches = list(INPUT_ROOT.rglob("custom_fold0.jsonl"))
if len(manifest_matches) != 1:
    available = [str(path.relative_to(INPUT_ROOT)) for path in INPUT_ROOT.rglob("*")]
    raise RuntimeError(f"expected one V2 manifest input, found {available}")
INPUT = manifest_matches[0].parents[2]
for item in INPUT.iterdir():
    if item.is_dir():
        shutil.copytree(item, WORKING / item.name)
    elif item.suffix in {".zip", ".tar"}:
        shutil.unpack_archive(item, WORKING)
    elif item.name != "dataset-metadata.json":
        shutil.copy2(item, WORKING / item.name)

if RUN_MODE == "validate":
    run(
        "python",
        "scripts/validate_ltr_cache_manifest.py",
        "--manifest",
        "data/manifests/custom_fold0.jsonl",
        "--split",
        "train",
    )
    print("V2 input validation passed. Attach the V2 pretrain Dataset before enabling five_fold.")
elif RUN_MODE == "five_fold":
    PRETRAINED = find_pretrained_checkpoint()
    for fold in range(5):
        run_fold(fold, PRETRAINED)
    shutil.make_archive("/kaggle/working/v2_tcn_ltr_outputs", "zip", WORKING / "data")
elif RUN_MODE == "fold":
    run_fold(FOLD_INDEX, find_pretrained_checkpoint())
    shutil.make_archive(f"/kaggle/working/v2_tcn_ltr_fold{FOLD_INDEX}_artifacts", "zip", WORKING / "data")
else:
    raise ValueError(f"unknown RUN_MODE: {RUN_MODE}")
