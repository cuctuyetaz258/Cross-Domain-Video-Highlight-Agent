import sys
from types import SimpleNamespace

import numpy as np

from highlight_agent.features.visual_new import extract_gesture_signal, extract_scene_changes


class _Timecode:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def get_seconds(self) -> float:
        return self.seconds


def test_scene_extract_empty_when_detector_fails(monkeypatch) -> None:
    def failing_detect(*args, **kwargs):
        raise RuntimeError("cannot read video")

    monkeypatch.setitem(
        sys.modules,
        "scenedetect",
        SimpleNamespace(ContentDetector=lambda **kwargs: object(), detect=failing_detect),
    )

    assert extract_scene_changes("missing.mp4", 20.0) == []


def test_scene_extract_returns_only_timestamps_inside_duration(monkeypatch) -> None:
    scenes = [(_Timecode(2.5), _Timecode(7.0)), (_Timecode(25.0), _Timecode(30.0))]
    monkeypatch.setitem(
        sys.modules,
        "scenedetect",
        SimpleNamespace(ContentDetector=lambda **kwargs: object(), detect=lambda *args, **kwargs: scenes),
    )

    assert extract_scene_changes("video.mp4", 10.0) == [2.5]


def test_gesture_shape_and_no_face_fallback(monkeypatch) -> None:
    class Capture:
        def isOpened(self) -> bool:
            return True

        def set(self, *args) -> None:
            return None

        def read(self):
            return True, object()

        def release(self) -> None:
            return None

    class FaceMesh:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def process(self, frame):
            return SimpleNamespace(multi_face_landmarks=None)

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda path: Capture(),
        CAP_PROP_POS_MSEC=0,
        COLOR_BGR2RGB=1,
        cvtColor=lambda frame, code: frame,
    )
    fake_mp = SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=FaceMesh)))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    signal = extract_gesture_signal("video.mp4", duration=3.0)

    assert signal.shape == (6,)
    assert signal.dtype == np.float32
    assert np.all(signal == 0.0)


def test_gesture_returns_zero_when_facemesh_cannot_initialize(monkeypatch) -> None:
    class Capture:
        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            return None

    class FaceMesh:
        def __init__(self, **kwargs) -> None:
            raise RuntimeError("gpu unavailable")

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(VideoCapture=lambda path: Capture()),
    )
    monkeypatch.setitem(
        sys.modules,
        "mediapipe",
        SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=FaceMesh))),
    )

    assert np.all(extract_gesture_signal("video.mp4", duration=1.0) == 0.0)


def test_gesture_uses_sequential_decode_when_fps_is_available(monkeypatch) -> None:
    class Capture:
        def __init__(self) -> None:
            self.grab_count = 0
            self.set_count = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop) -> float:
            return 4.0

        def grab(self) -> bool:
            self.grab_count += 1
            return True

        def retrieve(self):
            return True, object()

        def set(self, *args) -> None:
            self.set_count += 1

        def release(self) -> None:
            return None

    capture = Capture()

    class FaceMesh:
        def __init__(self, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def process(self, frame):
            return SimpleNamespace(multi_face_landmarks=None)

    fake_cv2 = SimpleNamespace(
        VideoCapture=lambda path: capture,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_MSEC=0,
        COLOR_BGR2RGB=1,
        cvtColor=lambda frame, code: frame,
    )
    fake_mp = SimpleNamespace(solutions=SimpleNamespace(face_mesh=SimpleNamespace(FaceMesh=FaceMesh)))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    signal = extract_gesture_signal("video.mp4", duration=2.0, sample_rate=2.0)

    assert signal.shape == (4,)
    assert capture.grab_count == 7
    assert capture.set_count == 0
