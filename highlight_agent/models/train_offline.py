"""Offline training utilities for the seven-channel LTR highlight scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .ltr_scorer import AdditiveAttentionScorer

FEATURE_SCHEMA_VERSION = "1.0"
FEATURE_SAMPLE_RATE = 10
FEATURE_CHANNELS = (
    "rms",
    "pitch",
    "silence",
    "text_score",
    "scene_change",
    "gesture",
    "turn_rate",
)
LABEL_TO_INT = {"negative": 0, "positive": 1, "ignored": -1}


@dataclass(frozen=True)
class WindowExample:
    """One chronological LTR window with provenance and a pooled feature vector."""

    video_id: str
    domain: str
    window_index: int
    start: float
    end: float
    feature: np.ndarray
    label: int
    score: float


def load_tvsum(mat_path: str | Path) -> list[dict[str, Any]]:
    """Load TVSum annotations into the common training record format."""

    import scipy.io

    data = scipy.io.loadmat(str(mat_path))["tvsum50"][0]
    records: list[dict[str, Any]] = []
    for row in data:
        video_id = str(row["video"][0])
        category = str(row["category"][0])
        frame_scores = row["annotations"].mean(axis=0).astype(np.float32)
        if frame_scores.ndim == 2 and frame_scores.shape[1] == 1:
            frame_scores = frame_scores.squeeze(1)
        domain = "lecture" if category == "LF" else "podcast" if category == "VT" else "standup"
        records.append(
            {
                "video_id": video_id,
                "domain": domain,
                "source": "tvsum",
                "frame_scores": frame_scores,
                "fps": 24.0,
            }
        )
    return records


def load_summe(gt_dir: str | Path) -> list[dict[str, Any]]:
    """Load SumMe ground-truth MAT files into the common record format."""

    import scipy.io

    records: list[dict[str, Any]] = []
    for path in sorted(Path(gt_dir).glob("*.mat")):
        mat = scipy.io.loadmat(str(path))
        records.append(
            {
                "video_id": path.stem,
                "domain": "standup",
                "source": "summe",
                "frame_scores": mat["gt_score"].squeeze().astype(np.float32),
                "fps": float(mat["fps"][0][0]) if "fps" in mat else 25.0,
            }
        )
    return records


def load_qvhighlights(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load QVHighlights JSONL annotations into the common record format."""

    records: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            relevant_windows = entry.get("relevant_windows", [])
            duration = float(
                entry.get("duration")
                or entry.get("video_duration")
                or max((window[1] for window in relevant_windows), default=0.0)
            )
            records.append(
                {
                    "video_id": entry["vid"],
                    "domain": "lecture",
                    "source": "qvhighlights",
                    "relevant_windows": relevant_windows,
                    "saliency_scores": entry.get("saliency_scores", []),
                    "duration": duration,
                }
            )
    return records


def load_training_manifest(
    jsonl_path: str | Path,
    *,
    split: str | None = None,
) -> list[dict[str, Any]]:
    """Load canonical project manifest records, optionally filtering by video split."""

    records: list[dict[str, Any]] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if "video_id" not in entry:
                raise ValueError(f"manifest line {line_number} is missing video_id")
            if split is not None and entry.get("split") != split:
                continue
            records.append(entry)
    return records


def create_window_labels(
    record: dict[str, Any],
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
) -> list[dict[str, Any]]:
    """Convert one dataset record into positive, negative and ignored windows."""

    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive")

    source = record.get("source")
    labels: list[dict[str, Any]] = []
    if source in {"tvsum", "summe"}:
        frame_scores = np.asarray(record["frame_scores"], dtype=np.float32).reshape(-1)
        fps = float(record["fps"])
        if fps <= 0:
            raise ValueError("record fps must be positive")
        target_len = int(len(frame_scores) / fps * FEATURE_SAMPLE_RATE)
        if target_len == 0:
            return []

        scores_10hz = np.interp(
            np.linspace(0, len(frame_scores) - 1, target_len),
            np.arange(len(frame_scores)),
            frame_scores,
        )
        window_frames = int(window_sec * FEATURE_SAMPLE_RATE)
        hop_frames = int(hop_sec * FEATURE_SAMPLE_RATE)
        q25 = np.percentile(scores_10hz, 25)
        q75 = np.percentile(scores_10hz, 75)
        for start_frame in range(0, len(scores_10hz), hop_frames):
            if start_frame + window_frames > len(scores_10hz):
                continue
            window_score = float(np.mean(scores_10hz[start_frame : start_frame + window_frames]))
            label = "positive" if window_score > q75 else "negative" if window_score < q25 else "ignored"
            labels.append(
                {
                    "start": start_frame / FEATURE_SAMPLE_RATE,
                    "end": (start_frame + window_frames) / FEATURE_SAMPLE_RATE,
                    "label": label,
                    "score": window_score,
                }
            )
        return labels

    if source in {"custom", "custom_pseudo"}:
        relevant_windows = record.get("relevant_windows", [])
        duration = float(
            record.get("duration")
            or max((window[1] for window in relevant_windows), default=0.0)
        )
        if duration <= 0:
            return []
        for start_sec in np.arange(0, duration, hop_sec):
            end_sec = float(start_sec + window_sec)
            if end_sec > duration:
                break
            max_coverage = 0.0
            for window_start, window_end in relevant_windows:
                intersection = max(
                    0.0,
                    min(end_sec, float(window_end)) - max(float(start_sec), float(window_start)),
                )
                max_coverage = max(max_coverage, intersection / window_sec)
            label = (
                "positive"
                if max_coverage >= 0.5
                else "negative"
                if max_coverage == 0.0
                else "ignored"
            )
            labels.append(
                {
                    "start": float(start_sec),
                    "end": end_sec,
                    "label": label,
                    "score": max_coverage,
                }
            )
        return labels

    if source != "qvhighlights":
        raise ValueError(f"unsupported training source: {source}")

    relevant_windows = record.get("relevant_windows", [])
    saliency_scores = record.get("saliency_scores", [])
    filtered_windows: list[list[float]] = []
    if saliency_scores:
        clip_max_saliency: dict[int, int] = {}
        for entry in saliency_scores:
            clip_index, _sentence_index, saliency = map(int, entry[:3])
            clip_max_saliency[clip_index] = max(clip_max_saliency.get(clip_index, 0), saliency)
        filtered_windows = [
            window
            for clip_index, window in enumerate(relevant_windows)
            if clip_max_saliency.get(clip_index, 0) >= 3
        ]
    if not filtered_windows:
        filtered_windows = relevant_windows

    duration = float(record.get("duration") or max((window[1] for window in filtered_windows), default=0.0))
    if duration <= 0:
        return []
    for start_sec in np.arange(0, duration, hop_sec):
        end_sec = float(start_sec + window_sec)
        if end_sec > duration:
            break
        max_iou = 0.0
        for window_start, window_end in filtered_windows:
            intersection = max(0.0, min(end_sec, window_end) - max(float(start_sec), window_start))
            union = max(end_sec, window_end) - min(float(start_sec), window_start)
            max_iou = max(max_iou, intersection / union if union > 0 else 0.0)
        label = "positive" if max_iou > 0.5 else "negative" if max_iou < 0.1 else "ignored"
        labels.append(
            {
                "start": float(start_sec),
                "end": end_sec,
                "label": label,
                "score": max_iou,
            }
        )
    return labels


def compute_lref(records: Iterable[dict[str, Any]]) -> float:
    """Compute the median reference highlight duration from annotations."""

    durations: list[float] = []
    for record in records:
        if record["source"] in {"qvhighlights", "custom", "custom_pseudo"}:
            durations.extend(float(end - start) for start, end in record.get("relevant_windows", []))
            continue
        frame_scores = np.asarray(record["frame_scores"], dtype=np.float32).reshape(-1)
        fps = float(record["fps"])
        binary = (frame_scores > np.percentile(frame_scores, 75)).astype(int)
        changes = np.diff(np.concatenate(([0], binary, [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        durations.extend(float(end - start) / fps for start, end in zip(starts, ends))

    positive_durations = [duration for duration in durations if duration > 0]
    return float(np.median(positive_durations)) if positive_durations else 40.0


def feature_cache_metadata(video_id: str, feature_matrix: np.ndarray) -> dict[str, Any]:
    """Build canonical metadata for a `(7, T)` feature cache file."""

    matrix = np.asarray(feature_matrix)
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "video_id": video_id,
        "sample_rate": FEATURE_SAMPLE_RATE,
        "channel_order": list(FEATURE_CHANNELS),
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }


def load_feature_matrix(feature_cache_dir: str | Path, video_id: str) -> np.ndarray:
    """Load and strictly validate one canonical feature cache."""

    cache_dir = Path(feature_cache_dir) / video_id
    feature_path = cache_dir / "feature_matrix.npy"
    metadata_path = cache_dir / "metadata.json"
    if not feature_path.is_file():
        raise FileNotFoundError(f"feature cache not found: {feature_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"feature cache metadata not found: {metadata_path}")

    matrix = np.load(feature_path, allow_pickle=False)
    if matrix.dtype != np.float32:
        raise ValueError(f"feature cache {video_id} must use float32, got {matrix.dtype}")
    if matrix.ndim != 2 or matrix.shape[0] != len(FEATURE_CHANNELS) or matrix.shape[1] == 0:
        raise ValueError(f"feature cache {video_id} must have shape (7, T), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"feature cache {video_id} contains non-finite values")
    if float(matrix.min()) < -1e-6 or float(matrix.max()) > 1.0 + 1e-6:
        raise ValueError(f"feature cache {video_id} values must be normalized to [0, 1]")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = feature_cache_metadata(video_id, matrix)
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"feature cache metadata mismatch for {video_id}: "
                f"{key}={metadata.get(key)!r}, expected {expected_value!r}"
            )
    return matrix


def build_window_examples(
    feature_cache_dir: str | Path,
    records: Iterable[dict[str, Any]],
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
) -> list[WindowExample]:
    """Pool canonical feature caches into chronological labeled windows."""

    examples: list[WindowExample] = []
    for record in records:
        video_id = str(record["video_id"])
        matrix = load_feature_matrix(feature_cache_dir, video_id)
        labels = create_window_labels(record, window_sec=window_sec, hop_sec=hop_sec)
        for window_index, label_record in enumerate(labels):
            start_frame = max(0, int(round(label_record["start"] * FEATURE_SAMPLE_RATE)))
            requested_end_frame = int(round(label_record["end"] * FEATURE_SAMPLE_RATE))
            if requested_end_frame > matrix.shape[1]:
                raise ValueError(
                    f"feature cache {video_id} ends before labeled window {window_index}"
                )
            end_frame = requested_end_frame
            if end_frame <= start_frame:
                raise ValueError(f"empty feature window for {video_id} at index {window_index}")
            feature = matrix[:, start_frame:end_frame].mean(axis=1).astype(np.float32)
            examples.append(
                WindowExample(
                    video_id=video_id,
                    domain=str(record.get("domain", "unknown")),
                    window_index=window_index,
                    start=float(label_record["start"]),
                    end=float(label_record["end"]),
                    feature=feature,
                    label=LABEL_TO_INT[label_record["label"]],
                    score=float(label_record["score"]),
                )
            )
    return examples


def _pair_examples(examples: Iterable[WindowExample]) -> list[tuple[WindowExample, WindowExample]]:
    by_video: dict[str, list[WindowExample]] = defaultdict(list)
    for example in examples:
        by_video[example.video_id].append(example)
    pairs: list[tuple[WindowExample, WindowExample]] = []
    for video_examples in by_video.values():
        positives = [example for example in video_examples if example.label == 1]
        negatives = [example for example in video_examples if example.label == 0]
        pairs.extend((positive, negative) for positive in positives for negative in negatives)
    return pairs


def create_pairwise_dataset(
    feature_cache_dir: str | Path,
    records: list[dict[str, Any]],
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create positive/negative feature pairs while preserving the legacy return type."""

    examples = build_window_examples(feature_cache_dir, records, window_sec, hop_sec)
    return [(positive.feature, negative.feature) for positive, negative in _pair_examples(examples)]


def margin_ranking_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Pairwise hinge loss requiring positive scores to exceed negatives by gamma."""

    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive_scores and negative_scores must have the same shape")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return torch.relu(gamma - positive_scores + negative_scores).mean()


def temporal_smoothness_loss(scores_by_video: Iterable[torch.Tensor]) -> torch.Tensor:
    """Mean adjacent-score penalty computed independently within each video."""

    sequences = [scores.reshape(-1) for scores in scores_by_video]
    valid = [scores for scores in sequences if scores.numel() >= 2]
    if not valid:
        device = sequences[0].device if sequences else torch.device("cpu")
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack([torch.mean(torch.diff(scores) ** 2) for scores in valid]).mean()


def evaluate_average_precision(
    model: torch.nn.Module,
    examples: Iterable[WindowExample],
    *,
    device: str | torch.device = "cpu",
) -> float:
    """Compute binary Average Precision over non-ignored windows."""

    labeled = [example for example in examples if example.label in {0, 1}]
    labels = np.asarray([example.label for example in labeled], dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("Average Precision requires both positive and negative validation windows")
    target_device = torch.device(device)
    model.to(target_device)
    features = torch.as_tensor(
        np.stack([example.feature for example in labeled]), dtype=torch.float32, device=target_device
    )
    model.eval()
    with torch.no_grad():
        scores = model(features).reshape(-1).detach().cpu().numpy()
    return float(average_precision_score(labels, scores))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _records_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item.get("video_id", ""))):
        summary = {
            "video_id": record.get("video_id"),
            "source": record.get("source"),
            "domain": record.get("domain"),
            "duration": record.get("duration"),
            "fps": record.get("fps"),
            "relevant_windows": record.get("relevant_windows"),
        }
        hasher.update(json.dumps(summary, sort_keys=True, default=str).encode("utf-8"))
        if "frame_scores" in record:
            hasher.update(np.asarray(record["frame_scores"], dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _write_training_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def train(
    feature_cache_dir: str | Path,
    records: list[dict[str, Any]],
    output_path: str | Path,
    val_records: list[dict[str, Any]] | None = None,
    hidden_dim: int = 32,
    gamma: float = 1.0,
    lambda_smooth: float = 0.01,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    window_sec: float = 5.0,
    hop_sec: float = 1.0,
    seed: int = 42,
    training_log_path: str | Path | None = None,
    last_checkpoint_path: str | Path | None = None,
) -> AdditiveAttentionScorer:
    """Train the scorer with ranking and temporal smoothness objectives."""

    if not records:
        raise ValueError("training records must not be empty")
    if hidden_dim <= 0 or batch_size <= 0 or max_epochs <= 0 or patience <= 0:
        raise ValueError("hidden_dim, batch_size, max_epochs and patience must be positive")
    if lambda_smooth < 0:
        raise ValueError("lambda_smooth must be non-negative")

    _set_seed(seed)
    output = Path(output_path)
    log_path = Path(training_log_path) if training_log_path else output.with_name("training_log.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_examples = build_window_examples(feature_cache_dir, records, window_sec, hop_sec)
    train_pairs = _pair_examples(train_examples)
    if not train_pairs:
        raise ValueError("training data must contain positive and negative windows in the same video")

    validation_examples = (
        build_window_examples(feature_cache_dir, val_records, window_sec, hop_sec) if val_records else []
    )
    selection_examples = validation_examples or train_examples
    selection_split = "validation" if validation_examples else "training"
    labeled_selection = [example.label for example in selection_examples if example.label in {0, 1}]
    if len(set(labeled_selection)) != 2:
        raise ValueError(f"{selection_split} data must contain positive and negative windows")

    chronological: dict[str, list[WindowExample]] = defaultdict(list)
    for example in train_examples:
        chronological[example.video_id].append(example)
    for video_id in chronological:
        chronological[video_id].sort(key=lambda example: example.window_index)

    model = AdditiveAttentionScorer(in_features=len(FEATURE_CHANNELS), hidden_dim=hidden_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    random_generator = random.Random(seed)
    l_ref = compute_lref(records)
    best_ap = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_logs: list[dict[str, Any]] = []
    config = {
        "hidden_dim": hidden_dim,
        "gamma": gamma,
        "lambda_smooth": lambda_smooth,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "window_sec": window_sec,
        "hop_sec": hop_sec,
        "seed": seed,
        "device": device.type,
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled_pairs = list(train_pairs)
        random_generator.shuffle(shuffled_pairs)
        margin_total = 0.0
        margin_items = 0
        for offset in range(0, len(shuffled_pairs), batch_size):
            batch = shuffled_pairs[offset : offset + batch_size]
            positive_features = torch.as_tensor(
                np.stack([positive.feature for positive, _ in batch]), dtype=torch.float32, device=device
            )
            negative_features = torch.as_tensor(
                np.stack([negative.feature for _, negative in batch]), dtype=torch.float32, device=device
            )
            optimizer.zero_grad()
            margin_loss = margin_ranking_loss(
                model(positive_features), model(negative_features), gamma=gamma
            )
            margin_loss.backward()
            optimizer.step()
            margin_total += float(margin_loss.detach().cpu()) * len(batch)
            margin_items += len(batch)

        smooth_values: list[float] = []
        if lambda_smooth > 0:
            for sequence in chronological.values():
                if len(sequence) < 2:
                    continue
                sequence_features = torch.as_tensor(
                    np.stack([example.feature for example in sequence]), dtype=torch.float32, device=device
                )
                optimizer.zero_grad()
                smooth_loss = temporal_smoothness_loss([model(sequence_features).reshape(-1)])
                (lambda_smooth * smooth_loss).backward()
                optimizer.step()
                smooth_values.append(float(smooth_loss.detach().cpu()))

        scheduler.step()
        margin_mean = margin_total / max(margin_items, 1)
        smooth_mean = float(np.mean(smooth_values)) if smooth_values else 0.0
        train_total = margin_mean + lambda_smooth * smooth_mean
        selection_ap = evaluate_average_precision(model, selection_examples, device=device)
        epoch_log: dict[str, Any] = {
            "epoch": epoch,
            "train_margin_loss": margin_mean,
            "train_smooth_loss": smooth_mean,
            "train_total_loss": train_total,
            "selection_ap": selection_ap,
            "selection_split": selection_split,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        epoch_log["val_ap" if selection_split == "validation" else "train_ap"] = selection_ap
        epoch_logs.append(epoch_log)

        if selection_ap > best_ap:
            best_ap = selection_ap
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save(
                output,
                metadata={
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_schema": {
                        "channel_order": list(FEATURE_CHANNELS),
                        "sample_rate": FEATURE_SAMPLE_RATE,
                        "window_sec": window_sec,
                        "hop_sec": hop_sec,
                    },
                    "L_ref": l_ref,
                    "epoch": best_epoch,
                    "selection_ap": best_ap,
                    "selection_split": selection_split,
                    "val_ap": best_ap if selection_split == "validation" else None,
                    "train_ap": best_ap if selection_split == "training" else None,
                    "dataset_fingerprint": _records_fingerprint(records),
                    "validation_fingerprint": (
                        _records_fingerprint(val_records) if val_records else None
                    ),
                    "config": config,
                },
            )
        else:
            epochs_without_improvement += 1

        if last_checkpoint_path is not None:
            model.save(
                last_checkpoint_path,
                metadata={
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_schema": {
                        "channel_order": list(FEATURE_CHANNELS),
                        "sample_rate": FEATURE_SAMPLE_RATE,
                        "window_sec": window_sec,
                        "hop_sec": hop_sec,
                    },
                    "L_ref": l_ref,
                    "epoch": epoch,
                    "selection_ap": selection_ap,
                    "selection_split": selection_split,
                    "val_ap": selection_ap if selection_split == "validation" else None,
                    "train_ap": selection_ap if selection_split == "training" else None,
                    "dataset_fingerprint": _records_fingerprint(records),
                    "validation_fingerprint": (
                        _records_fingerprint(val_records) if val_records else None
                    ),
                    "config": config,
                    "checkpoint_role": "last",
                },
            )

        _write_training_log(
            log_path,
            {
                "feature_schema": {
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "channel_order": list(FEATURE_CHANNELS),
                    "sample_rate": FEATURE_SAMPLE_RATE,
                },
                "config": config,
                "selection_split": selection_split,
                "best_epoch": best_epoch,
                "best_ap": best_ap,
                "epochs": epoch_logs,
            },
        )
        if epochs_without_improvement >= patience:
            break

    return AdditiveAttentionScorer.load(output, device="cpu")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tvsum", help="TVSum MAT file")
    parser.add_argument("--summe", help="SumMe MAT directory")
    parser.add_argument("--qvhighlights", help="QVHighlights train JSONL")
    parser.add_argument("--val-qvhighlights", help="QVHighlights validation JSONL")
    parser.add_argument("--manifest", help="Canonical project training manifest JSONL")
    parser.add_argument("--train-split", default="train", help="Manifest training split")
    parser.add_argument("--val-split", default="val", help="Manifest validation split")
    parser.add_argument("--cache-dir", required=True, help="Canonical feature cache directory")
    parser.add_argument("--output", required=True, help="Output checkpoint path")
    parser.add_argument("--training-log", default=None, help="Optional training log JSON path")
    parser.add_argument("--last-output", default=None, help="Optional last-epoch checkpoint path")
    parser.add_argument("--training-config", default=None, help="Optional resolved config JSON path")
    parser.add_argument("--evaluation-snapshot", default=None, help="Optional metric snapshot JSON path")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records: list[dict[str, Any]] = []
    if args.tvsum:
        records.extend(load_tvsum(args.tvsum))
    if args.summe:
        records.extend(load_summe(args.summe))
    if args.qvhighlights:
        records.extend(load_qvhighlights(args.qvhighlights))
    validation_records = load_qvhighlights(args.val_qvhighlights) if args.val_qvhighlights else None
    if args.manifest:
        records.extend(load_training_manifest(args.manifest, split=args.train_split))
        manifest_validation = load_training_manifest(args.manifest, split=args.val_split)
        validation_records = (validation_records or []) + manifest_validation
    model = train(
        args.cache_dir,
        records,
        args.output,
        validation_records,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        lambda_smooth=args.lambda_smooth,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        seed=args.seed,
        training_log_path=args.training_log,
        last_checkpoint_path=args.last_output,
    )
    resolved_config = {
        "manifest": args.manifest,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "cache_dir": args.cache_dir,
        "output": args.output,
        "last_output": args.last_output,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "lambda_smooth": args.lambda_smooth,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "seed": args.seed,
    }
    if args.training_config:
        _write_training_log(Path(args.training_config), resolved_config)
    if args.evaluation_snapshot:
        train_examples = build_window_examples(
            args.cache_dir, records, args.window_sec, args.hop_sec
        )
        train_ap = evaluate_average_precision(model, train_examples)
        snapshot: dict[str, Any] = {
            "train_video_count": len({record["video_id"] for record in records}),
            "train_window_count": len(train_examples),
            "train_ap": train_ap,
            "validation_video_count": 0,
            "validation_window_count": 0,
            "val_ap": None,
        }
        if validation_records:
            validation_examples = build_window_examples(
                args.cache_dir, validation_records, args.window_sec, args.hop_sec
            )
            snapshot.update(
                {
                    "validation_video_count": len(
                        {record["video_id"] for record in validation_records}
                    ),
                    "validation_window_count": len(validation_examples),
                    "val_ap": evaluate_average_precision(model, validation_examples),
                }
            )
        _write_training_log(Path(args.evaluation_snapshot), snapshot)


if __name__ == "__main__":
    main()
