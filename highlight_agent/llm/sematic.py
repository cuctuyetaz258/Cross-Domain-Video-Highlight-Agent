# sematic score tu transcript
from IPython.display import display
import os
import json
import numpy as np
import pandas as pd
import librosa
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer, util

semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_semantic_scores(segments: List[str]) -> List[float]:
    """
    Tinh Semantic Score cho tung doan transcript bang Sentence-Transformers.
    Cach lam: embed tung cau/doan + embed vector ngu canh chung (trung binh toan bo),
    diem so = cosine similarity giua doan do va ngu canh chung
    -> doan cang 'trung tam / dai dien' cho noi dung thi diem cang cao.
    """
    if not segments:
        return []
    seg_embeddings = semantic_model.encode(segments, convert_to_tensor=True)
    doc_embedding = seg_embeddings.mean(dim=0, keepdim=True)
    sims = util.cos_sim(seg_embeddings, doc_embedding).squeeze(1)
    scores = sims.cpu().numpy().tolist()
    # normalize ve [0, 1]
    lo, hi = min(scores), max(scores)
    if hi - lo > 1e-9:
        scores = [(s - lo) / (hi - lo) for s in scores]
    return scores

# phuong an thay the dung LLm
def compute_semantic_score_llm(segments: List[str], api_call_fn) -> List[float]:
    scores = []
    for seg in segments:
        prompt = (
            f'"{seg}"'
        )
        response = api_call_fn(prompt)
        try:
            scores.append(float(response.strip()))
        except ValueError:
            scores.append(0.5)  # fallback neu LLM tra loi khong parse duoc
    return scores


print("Semantic Score module (Sentence-Transformers + LLM fallback) da san sang.")

from sklearn.feature_extraction.text import TfidfVectorizer


def compute_keyword_importance(segments: List[str], top_k: int = 5) -> List[dict]:
    """
    Tinh diem quan trong tu khoa cho tung doan bang TF-IDF tren toan bo transcript.
    Tra ve: diem tong hop (trung binh TF-IDF cua top-k tu) + danh sach tu khoa noi bat.
    """
    if not segments:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
    tfidf_matrix = vectorizer.fit_transform(segments)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results = []
    for i in range(len(segments)):
        row = tfidf_matrix[i].toarray().flatten()
        top_idx = row.argsort()[::-1][:top_k]
        top_idx = [idx for idx in top_idx if row[idx] > 0]
        keywords = feature_names[top_idx].tolist() if len(top_idx) else []
        keyword_score = float(row[top_idx].mean()) if len(top_idx) else 0.0
        results.append({"keyword_score": keyword_score, "top_keywords": keywords})
    return results


print("Keyword Importance module (TF-IDF) da san sang.")
print("Luu y: transcript Tieng Viet nen dung stop_words rieng thay vi \'english\' - co the thay bang danh sach stopwords VN.")
#  speaker change
from dataclasses import dataclass, field
@dataclass
class TranscriptTurn:
    speaker: str
    start: float
    end: float
    text: str = ""


def count_speaker_changes(turns: List[TranscriptTurn], window_start: float, window_end: float) -> dict:
    """
    Voi danh sach cac luot noi (lay tu pyannote/speaker-diarization-3.1) trong 1 cua so thoi gian:
    - Dem so lan doi speaker (turn changes)
    - Tinh turn-taking rate = so luot noi / phut
    """
    in_window = [t for t in turns if t.end > window_start and t.start < window_end]
    in_window.sort(key=lambda t: t.start)

    changes = 0
    for i in range(1, len(in_window)):
        if in_window[i].speaker != in_window[i - 1].speaker:
            changes += 1

    duration_min = max((window_end - window_start) / 60.0, 1e-6)
    turn_taking_rate = len(in_window) / duration_min  # so luot noi / phut

    return {
        "num_turns": len(in_window),
        "speaker_changes": changes,
        "turn_taking_rate_per_min": turn_taking_rate,
        "unique_speakers": len(set(t.speaker for t in in_window)),
    }


def interaction_score_from_turns(turns: List[TranscriptTurn], window_start: float,
                                  window_end: float, max_rate_ref: float = 12.0) -> float:
    """
    Chuan hoa turn-taking rate ve [0,1] lam diem Interaction Score cho tang 'Tuong tac'.
    max_rate_ref: nguong tham chieu (dieu chinh sau khi co du lieu podcast that).
    """
    stats = count_speaker_changes(turns, window_start, window_end)
    return min(stats["turn_taking_rate_per_min"] / max_rate_ref, 1.0)


print("Speaker turn-taking module da san sang.")
# demo voi dl gia lap

demo_segments = [
    "Chao mung cac ban den voi podcast hom nay, chung ta se noi ve AI.",
    "Day la mot khoanh khac rat quan trong trong lich su phat trien machine learning.",
    "A thi, um, toi nghi la, khong co gi dac biet lam dau.",
    "Diem mau chot o day la mo hinh Transformer da thay doi hoan toan cach chung ta xu ly ngon ngu.",
]

semantic_scores = compute_semantic_scores(demo_segments)
keyword_info = compute_keyword_importance(demo_segments)

demo_turns = [
    TranscriptTurn("A", 0, 8, demo_segments[0]),
    TranscriptTurn("B", 8, 20, demo_segments[1]),
    TranscriptTurn("A", 20, 25, demo_segments[2]),
    TranscriptTurn("B", 25, 40, demo_segments[3]),
]
interaction_stats = count_speaker_changes(demo_turns, 0, 40)

demo_df = pd.DataFrame({
    "segment": demo_segments,
    "semantic_score": semantic_scores,
    "keyword_score": [k["keyword_score"] for k in keyword_info],
    "top_keywords": [k["top_keywords"] for k in keyword_info],
})

display(demo_df)

print(interaction_stats)

print(interaction_score_from_turns(demo_turns, 0, 40))