import os
import json
import numpy as np
import pandas as pd
import librosa
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# khai bao 5 tang dac trung
LAYERS = [
    "acoustic",        # Am hoc
    "paralinguistic",  # Can ngon ngu
    "linguistic",      # Ngon ngu
    "structural",      # Cau truc
    "interaction",      # Tuong tac
]

@dataclass
class FeatureWindow:
    """Mot cua so thoi gian (mac dinh 30s) chua diem so cua 5 tang dac trung."""
    start: float
    end: float
    scores: Dict[str, float] = field(default_factory=lambda: {l: 0.0 for l in LAYERS})
    meta: Dict = field(default_factory=dict)

    def total_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        w = weights or {l: 1.0 / len(LAYERS) for l in LAYERS}
        return sum(self.scores.get(l, 0.0) * w.get(l, 0.0) for l in LAYERS)


class LayerExtractor:
    """Base class cho tung tang dac trung -- moi tang cu the se override extract()."""
    name: str = "base"

    def extract(self, window_data: dict) -> float:
        raise NotImplementedError


class AcousticExtractor(LayerExtractor):
    name = "acoustic"
    # TODO (Khanh Van, buoi 2): RMS energy, pitch variation, silence duration (librosa)


class ParalinguisticExtractor(LayerExtractor):
    name = "paralinguistic"
    # TODO: laughter detection, emotion (wav2vec2 / distilbert)


class LinguisticExtractor(LayerExtractor):
    name = "linguistic"
    # -> Semantic score + keyword importance, xem Phan 2 ben duoi (Tat Nguyen)


class StructuralExtractor(LayerExtractor):
    name = "structural"
    # TODO: scene change, slide transition (OpenCV / PySceneDetect)


class InteractionExtractor(LayerExtractor):
    name = "interaction"
    # -> Speaker changes, turn-taking rate, xem Phan 2 ben duoi (Tat Nguyen)


EXTRACTORS = {
    "acoustic": AcousticExtractor(),
    "paralinguistic": ParalinguisticExtractor(),
    "linguistic": LinguisticExtractor(),
    "structural": StructuralExtractor(),
    "interaction": InteractionExtractor(),
}

print("Khung 5 tang dac trung da san sang:", LAYERS)
# cau hinh librosa

SAMPLE_RATE = 16000       # chuan hoa 16kHz mono (khop voi ffmpeg pipeline cua nhom)
WINDOW_SEC = 30.0         # kich thuoc cua so truot (theo de cuong)
HOP_SEC = 15.0            # buoc truot (50% overlap)


def load_audio(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load 1 file audio (.wav/.mp3) ve mono, sample rate chuan."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def make_sliding_windows(duration_sec: float, window_sec: float = WINDOW_SEC,
                          hop_sec: float = HOP_SEC) -> List[FeatureWindow]:
    """Sinh danh sach cua so thoi gian truot tren toan bo audio/video."""
    windows = []
    t = 0.0
    while t < duration_sec:
        end = min(t + window_sec, duration_sec)
        windows.append(FeatureWindow(start=t, end=end))
        if end >= duration_sec:
            break
        t += hop_sec
    return windows


def audio_segment(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Cat doan audio theo thoi gian (giay)."""
    return y[int(start * sr): int(end * sr)]


def run_pipeline(y: np.ndarray, sr: int, extractors: Dict[str, LayerExtractor] = None) -> pd.DataFrame:
    """Chay toan bo 5 tang tren tung cua so, tra ve DataFrame diem so theo thoi gian."""
    extractors = extractors or EXTRACTORS
    duration = len(y) / sr
    windows = make_sliding_windows(duration)
    rows = []
    for w in windows:
        seg = audio_segment(y, sr, w.start, w.end)
        row = {"start": w.start, "end": w.end}
        for name, extractor in extractors.items():
            try:
                row[name] = extractor.extract({"audio": seg, "sr": sr})
            except NotImplementedError:
                row[name] = None
        rows.append(row)
    return pd.DataFrame(rows)


print(f"Sliding window config: window={WINDOW_SEC}s, hop={HOP_SEC}s, sr={SAMPLE_RATE}Hz")
# thu thap data
import subprocess

SAMPLE_DIR = "sample_dataset"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Dien URL YouTube mau cho tung mien (thay bang video that cua nhom: 5-30 phut,
# theo de cuong can 10-16 video cho 2 mien Lecture + Podcast)
SAMPLE_VIDEOS = {
    "lecture_01": "https://www.youtube.com/watch?v=XXXXXXXXXXX",
    "podcast_01": "https://www.youtube.com/watch?v=YYYYYYYYYYY",
}


def download_sample(name: str, url: str, out_dir: str = SAMPLE_DIR):
    """Tai audio (.wav 16kHz mono) + phu de tu dong (neu co) bang yt-dlp."""
    out_path = os.path.join(out_dir, name)
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "--postprocessor-args", f"-ar {SAMPLE_RATE} -ac 1",
        "--write-auto-sub", "--sub-lang", "en,vi",
        "-o", f"{out_path}.%(ext)s",
        url,
    ]
    subprocess.run(cmd, check=False)
    print(f"Da tai xong: {name}")


print("Khung thu thap du lieu mau da san sang.")
print("-> Dien URL that vao SAMPLE_VIDEOS roi bo comment vong for de tai.")