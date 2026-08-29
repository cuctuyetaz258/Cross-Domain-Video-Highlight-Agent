"""Tiện ích xử lý đường dẫn portable trong manifest của project"""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve một đường dẫn relative của manifest trên mọi hệ điều hành"""

    raw_value = str(value)
    windows_path = PureWindowsPath(raw_value)
    posix_path = PurePosixPath(raw_value)
    if (
        Path(raw_value).is_absolute()
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"manifest path must be relative to project root: {value}")

    root = Path(project_root).resolve()
    relative_path = Path(*raw_value.replace("\\", "/").split("/"))
    if ".." in relative_path.parts:
        raise ValueError(f"manifest path must not escape project root: {value}")
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest path must stay inside project root: {value}") from exc
    return resolved


def portable_relative_path(path: str | Path, project_root: str | Path) -> str:
    """Serialize một path dưới project root bằng separator `/` ổn định"""

    root = Path(project_root).resolve()
    return Path(path).resolve().relative_to(root).as_posix()
