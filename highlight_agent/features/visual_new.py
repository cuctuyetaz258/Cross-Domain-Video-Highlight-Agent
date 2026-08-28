"""Trích xuất scene change và gesture cho LTR feature matrix"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GestureExtraction:
    """Kết quả gesture kèm trạng thái để phân biệt fallback và không có mặt"""

    signal: np.ndarray
    status: str
    decoded_sample_count: int
    detected_sample_count: int


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

    return extract_gesture_observation(video_path, duration, sample_rate).signal


def extract_gesture_observation(
    video_path: str | Path,
    duration: float,
    sample_rate: float = 2.0,
) -> GestureExtraction:
    """Trả gesture cùng trạng thái để nhận biết lỗi khởi tạo FaceMesh"""

    if duration <= 0 or sample_rate <= 0:
        raise ValueError("duration and sample_rate must be positive")

    sample_count = int(duration * sample_rate)
    signal = np.zeros(sample_count, dtype=np.float32)
    if sample_count == 0:
        return GestureExtraction(signal, "no_samples", 0, 0)

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("Vui lòng cài scenedetect và mediapipe cho LTR visual features") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return GestureExtraction(signal, "video_unreadable", 0, 0)

    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=False,
        )
    except Exception:
        capture.release()
        return GestureExtraction(signal, "facemesh_initialization_failed", 0, 0)

    decoded_sample_count = 0
    detected_sample_count = 0
    try:
        with face_mesh:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) if hasattr(capture, "get") else 0.0
            can_decode_sequentially = (
                fps > 0
                and hasattr(capture, "grab")
                and hasattr(capture, "retrieve")
            )
            current_frame = 0
            for index in range(sample_count):
                if can_decode_sequentially:
                    target_frame = int(round(index * fps / sample_rate))
                    grabbed = True
                    while current_frame <= target_frame:
                        grabbed = bool(capture.grab())
                        current_frame += 1
                        if not grabbed:
                            break
                    success, frame = capture.retrieve() if grabbed else (False, None)
                else:
                    capture.set(cv2.CAP_PROP_POS_MSEC, index * 1000.0 / sample_rate)
                    success, frame = capture.read()
                if not success or frame is None:
                    continue
                decoded_sample_count += 1
                result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not result.multi_face_landmarks:
                    continue
                landmarks = result.multi_face_landmarks[0].landmark
                signal[index] = abs(landmarks[13].y - landmarks[14].y)
                detected_sample_count += 1
    finally:
        capture.release()
    if decoded_sample_count == 0:
        status = "no_decodable_samples"
    elif detected_sample_count == 0:
        status = "no_face_detected"
    elif detected_sample_count == decoded_sample_count:
        status = "ok"
    else:
        status = "partial_face_detection"
    return GestureExtraction(signal, status, decoded_sample_count, detected_sample_count)
