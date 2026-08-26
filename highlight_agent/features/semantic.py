"""Trích xuất evidence ngữ nghĩa từ transcript theo từng cửa sổ"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from highlight_agent.schemas import SemanticFeatures, TranscriptDocument

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_CUE_PHRASES = (
    "the key point",
    "the important thing",
    "the main point",
    "in summary",
    "to summarize",
    "the takeaway",
    "for example",
    "the reason is",
    "what matters",
    "in conclusion",
    "điều quan trọng",
    "điểm chính",
    "tóm lại",
    "ví dụ",
    "kết luận",
)


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class SemanticWindowScore:
    """Evidence semantic thô của một cửa sổ"""

    start: float
    end: float
    features: SemanticFeatures


class _SemanticModelLoader:
    """Nạp MiniLM khi semantic layer được gọi lần đầu"""

    _model: SentenceEncoder | None = None

    @classmethod
    def get_model(cls) -> SentenceEncoder:
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(DEFAULT_SEMANTIC_MODEL, device="cpu")
        return cls._model


def _window_text(document: TranscriptDocument, start: float, end: float) -> tuple[str, float]:
    parts: list[str] = []
    covered_seconds = 0.0
    for segment in document.segments:
        overlap = max(0.0, min(end, segment.end) - max(start, segment.start))
        if overlap <= 0:
            continue
        parts.append(segment.text)
        covered_seconds += overlap
    coverage = min(1.0, covered_seconds / (end - start)) if end > start else 0.0
    return " ".join(parts), coverage


def _cue_matches(text: str) -> list[str]:
    lowered = text.casefold()
    return [phrase for phrase in _CUE_PHRASES if phrase in lowered]


def _tfidf_density(texts: list[str]) -> np.ndarray:
    non_empty_indices = [index for index, text in enumerate(texts) if text.strip()]
    result = np.zeros(len(texts), dtype=np.float32)
    if not non_empty_indices:
        return result
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [texts[index] for index in non_empty_indices]
        )
    except ValueError:
        return result
    densities = np.asarray(matrix.mean(axis=1)).ravel()
    # Mean TF-IDF rất nhỏ với transcript dài nên co giãn về thang evidence 0-1
    result[non_empty_indices] = np.clip(densities * 10.0, 0.0, 1.0)
    return result


def extract_windowed_semantic_features(
    transcript: TranscriptDocument,
    *,
    window_seconds: float = 30.0,
    hop_seconds: float = 30.0,
    encoder: SentenceEncoder | None = None,
    duration: float | None = None,
) -> list[SemanticWindowScore]:
    """Tạo semantic score theo cửa sổ từ embedding, TF-IDF và cue phrase"""

    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive when provided")

    analysis_duration = duration if duration is not None else transcript.duration
    starts = list(np.arange(0.0, analysis_duration, hop_seconds))
    windows = [(float(start), float(min(start + window_seconds, analysis_duration))) for start in starts]
    texts_and_coverage = [_window_text(transcript, start, end) for start, end in windows]
    texts = [item[0] for item in texts_and_coverage]
    coverage = [item[1] for item in texts_and_coverage]
    tfidf = _tfidf_density(texts)

    embeddings = np.zeros((len(windows), 1), dtype=np.float32)
    non_empty = [index for index, text in enumerate(texts) if text.strip()]
    if non_empty:
        active_encoder = encoder or _SemanticModelLoader.get_model()
        encoded = np.asarray(
            active_encoder.encode(
                [texts[index] for index in non_empty],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        embeddings = np.zeros((len(windows), encoded.shape[1]), dtype=np.float32)
        embeddings[non_empty] = encoded

    topic = np.zeros(len(windows), dtype=np.float32)
    novelty = np.zeros(len(windows), dtype=np.float32)
    if non_empty:
        centroid = embeddings[non_empty].mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid /= centroid_norm
            topic[non_empty] = np.clip((embeddings[non_empty] @ centroid + 1.0) / 2.0, 0.0, 1.0)
        for index in non_empty:
            previous = [item for item in non_empty if max(0, index - 3) <= item < index]
            if previous:
                similarity = float(np.max(embeddings[previous] @ embeddings[index]))
                novelty[index] = np.clip(1.0 - similarity, 0.0, 1.0)

    results: list[SemanticWindowScore] = []
    for index, (start, end) in enumerate(windows):
        cues = _cue_matches(texts[index])
        cue_score = min(1.0, len(cues) / 2.0)
        raw_score = float(
            0.40 * topic[index]
            + 0.25 * novelty[index]
            + 0.20 * tfidf[index]
            + 0.15 * cue_score
        )
        results.append(
            SemanticWindowScore(
                start=round(start, 3),
                end=round(end, 3),
                features=SemanticFeatures(
                    topic_relevance=round(float(topic[index]), 6),
                    semantic_novelty=round(float(novelty[index]), 6),
                    tfidf_density=round(float(tfidf[index]), 6),
                    cue_score=round(float(cue_score), 6),
                    raw_score=round(raw_score, 6),
                    cue_phrases=cues,
                    text_coverage=round(float(coverage[index]), 6),
                ),
            )
        )
    return results
