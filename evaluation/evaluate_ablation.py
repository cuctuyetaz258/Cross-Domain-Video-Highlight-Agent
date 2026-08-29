"""Compatibility entry point for the checkpoint-backed LTR variant evaluation.

The former ablation script evaluated legacy visual/heuristic pipelines and silently
substituted random predictions when artifacts were missing. The production
pipeline is checkpoint-required, so ablation reporting now delegates to the
authoritative seven-channel evaluator. Missing checkpoints, caches, manifests,
or LLM run metadata remain explicit errors or ``not_run`` results.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate_ltr_variants import main  # noqa: E402

if __name__ == "__main__":
    main()
