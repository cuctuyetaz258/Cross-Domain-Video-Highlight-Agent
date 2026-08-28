"""Trích xuất scene change và gesture cho LTR feature matrix"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_scene_changes(
    video_path: str | Path,
    duration: float,
    *,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[float]:
    """Trả timestamp scene cut, lỗi đọc video thì trả mảng rỗng"""

    if duration <= 0:
        raise ValueError("duration must be positive")
    if threshold <= 0 or min_scene_len < 1:
        raise ValueError("threshold must be positive and min_scene_len must be at least one")

    try:
        from scenedetect import ContentDetector, detect

        scenes = detect(
            str(video_path),
            ContentDetector(threshold=threshold, min_scene_len=min_scene_len),
        )
    except Exception:
        return []

    timestamps = [float(start.get_seconds()) for start, _ in scenes]
    return [timestamp for timestamp in timestamps if 0.0 <= timestamp <= duration]


def extract_gesture_signal(
    video_path: str | Path,
    duration: float,
    sample_rate: float = 2.0,
) -> np.ndarray:
    """Đo độ mở miệng FaceMesh tại các mốc lấy mẫu thưa"""

    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")

    sample_count = int(duration * sample_rate)
    signal = np.zeros(sample_count, dtype=np.float32)
    if sample_count == 0:
        return signal

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("Vui lòng cài scenedetect và mediapipe cho LTR visual features") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return signal

    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=False,
        )
    except Exception:
        capture.release()
        return signal

    try:
        with face_mesh:
            for index in range(sample_count):
                capture.set(cv2.CAP_PROP_POS_MSEC, index * 1000.0 / sample_rate)
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.multi_face_landmarks:
                    continue
                landmarks = result.multi_face_landmarks[0].landmark
                signal[index] = abs(landmarks[13].y - landmarks[14].y)
    finally:
        capture.release()
    return signal
