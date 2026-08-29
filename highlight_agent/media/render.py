"""Render clip video (9:16 hoặc 16:9), phụ đề, thumbnail và metadata"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from highlight_agent.schemas import (
    BoundaryAdjustment,
    HighlightCandidate,
    LLMHighlightAssessment,
    MediaWorkspace,
    RenderedHighlight,
    TranscriptDocument,
)

from .audio import probe_duration, require_executable, run_media_command
from .errors import MediaProcessingError

AspectRatio = Literal["9:16", "16:9"]
DEFAULT_ASPECT_RATIO: AspectRatio = "9:16"


@dataclass(frozen=True)
class VideoFormatSpec:
    width: int
    height: int
    thumbnail_width: int
    thumbnail_height: int
    subtitle_font_size: int
    subtitle_margin_v: int
    label: str


VIDEO_FORMAT_SPECS: dict[AspectRatio, VideoFormatSpec] = {
    "9:16": VideoFormatSpec(
        width=1080,
        height=1920,
        thumbnail_width=720,
        thumbnail_height=1280,
        subtitle_font_size=18,
        subtitle_margin_v=80,
        label="Portrait (9:16)",
    ),
    "16:9": VideoFormatSpec(
        width=1920,
        height=1080,
        thumbnail_width=1280,
        thumbnail_height=720,
        subtitle_font_size=22,
        subtitle_margin_v=45,
        label="Landscape (16:9)",
    ),
}


def get_video_format_spec(aspect_ratio: str = DEFAULT_ASPECT_RATIO) -> VideoFormatSpec:
    if aspect_ratio not in VIDEO_FORMAT_SPECS:
        valid_options = ", ".join(sorted(VIDEO_FORMAT_SPECS.keys()))
        raise ValueError(f"unsupported aspect ratio '{aspect_ratio}', must be one of: {valid_options}")
    return VIDEO_FORMAT_SPECS[aspect_ratio]  # type: ignore[index]


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_highlight_srt(
    transcript: TranscriptDocument,
    candidate: HighlightCandidate,
    output_path: str | Path,
) -> Path | None:
    """Tạo phụ đề segment theo mốc bắt đầu của highlight"""

    entries: list[str] = []
    for index, segment in enumerate(transcript.segments):
        display_end = segment.end
        if index + 1 < len(transcript.segments):
            display_end = min(display_end, transcript.segments[index + 1].start)

        overlap_start = max(segment.start, candidate.start_time)
        overlap_end = min(display_end, candidate.end_time)
        if overlap_end <= overlap_start:
            continue
        relative_start = overlap_start - candidate.start_time
        relative_end = overlap_end - candidate.start_time
        entries.append(
            f"{len(entries) + 1}\n"
            f"{_srt_timestamp(relative_start)} --> {_srt_timestamp(relative_end)}\n"
            f"{segment.text}\n"
        )

    if not entries:
        return None
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries), encoding="utf-8")
    return path


def _escape_ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def render_short_video(
    source_video: str | Path,
    candidate: HighlightCandidate,
    output_path: str | Path,
    *,
    aspect_ratio: AspectRatio = DEFAULT_ASPECT_RATIO,
    source_duration: float | None = None,
    subtitle_path: str | Path | None = None,
) -> Path:
    source = Path(source_video)
    if not source.is_file():
        raise MediaProcessingError(f"source video does not exist: {source}")

    duration = source_duration or probe_duration(source)
    if candidate.end_time > duration + 0.05:
        raise MediaProcessingError(
            f"highlight '{candidate.candidate_id}' ends at {candidate.end_time}s "
            f"but video duration is {duration:.2f}s"
        )

    spec = get_video_format_spec(aspect_ratio)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        f"scale={spec.width}:{spec.height}:force_original_aspect_ratio=increase",
        f"crop={spec.width}:{spec.height}",
        "setsar=1",
    ]
    if subtitle_path:
        escaped_path = _escape_ffmpeg_filter_path(Path(subtitle_path))
        filters.append(
            "subtitles="
            f"filename='{escaped_path}':"
            f"force_style='FontName=Arial,FontSize={spec.subtitle_font_size},Outline=2,Shadow=1,Alignment=2,MarginV={spec.subtitle_margin_v}'"
        )

    ffmpeg = require_executable("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-ss",
        str(candidate.start_time),
        "-i",
        str(source),
        "-t",
        str(candidate.end_time - candidate.start_time),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        ",".join(filters),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    run_media_command(command, f"rendering highlight '{candidate.candidate_id}'")
    return output


def render_short_9_16(
    source_video: str | Path,
    candidate: HighlightCandidate,
    output_path: str | Path,
    *,
    source_duration: float | None = None,
    subtitle_path: str | Path | None = None,
) -> Path:
    """Wrapper tương thích ngược cho việc render clip 9:16"""
    return render_short_video(
        source_video,
        candidate,
        output_path,
        aspect_ratio="9:16",
        source_duration=source_duration,
        subtitle_path=subtitle_path,
    )


def extract_thumbnail(
    source_video: str | Path,
    candidate: HighlightCandidate,
    output_path: str | Path,
    *,
    aspect_ratio: AspectRatio = DEFAULT_ASPECT_RATIO,
) -> Path:
    spec = get_video_format_spec(aspect_ratio)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    midpoint = candidate.start_time + (candidate.end_time - candidate.start_time) / 2
    ffmpeg = require_executable("ffmpeg")
    command = [
        ffmpeg,
        "-y",
        "-ss",
        str(midpoint),
        "-i",
        str(source_video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={spec.thumbnail_width}:{spec.thumbnail_height}:force_original_aspect_ratio=increase,crop={spec.thumbnail_width}:{spec.thumbnail_height}",
        "-q:v",
        "2",
        str(output),
    ]
    run_media_command(command, f"extracting thumbnail for '{candidate.candidate_id}'")
    return output


def render_highlights(
    workspace: MediaWorkspace,
    candidates: list[HighlightCandidate],
    *,
    aspect_ratio: AspectRatio = DEFAULT_ASPECT_RATIO,
    transcript: TranscriptDocument | None = None,
    burn_subtitles: bool = True,
    boundary_adjustments: list[BoundaryAdjustment] | None = None,
    llm_assessments: dict[str, LLMHighlightAssessment] | None = None,
    pipeline_metadata: dict | None = None,
) -> list[RenderedHighlight]:
    if not 3 <= len(candidates) <= 5:
        raise ValueError("MVP rendering expects between 3 and 5 highlight candidates")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("highlight candidate IDs must be unique")
    if transcript and transcript.video_id != workspace.video_id:
        raise ValueError("transcript video_id must match workspace video_id")
    if boundary_adjustments and {item.candidate_id for item in boundary_adjustments} != {
        candidate.candidate_id for candidate in candidates
    }:
        raise ValueError("boundary adjustments must match rendered candidate IDs")
    unknown_assessment_ids = set(llm_assessments or {}) - {
        candidate.candidate_id for candidate in candidates
    }
    if unknown_assessment_ids:
        raise ValueError(f"LLM assessments contain unknown rendered candidate IDs: {sorted(unknown_assessment_ids)}")

    spec = get_video_format_spec(aspect_ratio)
    workspace_dir = workspace.transcript_path.parent
    shorts_dir = workspace_dir / "shorts"
    thumbnails_dir = workspace_dir / "thumbnails"
    shorts_dir.mkdir(parents=True, exist_ok=True)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    source_duration = probe_duration(workspace.source_video_path)

    rendered: list[RenderedHighlight] = []
    for index, candidate in enumerate(candidates, start=1):
        basename = f"highlight_{index:02d}"
        subtitle_path = None
        if transcript and burn_subtitles:
            subtitle_path = write_highlight_srt(transcript, candidate, shorts_dir / f"{basename}.srt")

        video_path = render_short_video(
            workspace.source_video_path,
            candidate,
            shorts_dir / f"{basename}.mp4",
            aspect_ratio=aspect_ratio,
            source_duration=source_duration,
            subtitle_path=subtitle_path,
        )
        thumbnail_path = extract_thumbnail(
            workspace.source_video_path,
            candidate,
            thumbnails_dir / f"{basename}.jpg",
            aspect_ratio=aspect_ratio,
        )
        assessment = (llm_assessments or {}).get(candidate.candidate_id)
        rendered.append(
            RenderedHighlight(
                candidate_id=candidate.candidate_id,
                video_path=video_path,
                thumbnail_path=thumbnail_path,
                start_time=candidate.start_time,
                end_time=candidate.end_time,
                reason=candidate.reason,
                aspect_ratio=aspect_ratio,
                width=spec.width,
                height=spec.height,
                title=assessment.title if assessment else None,
                summary=assessment.summary if assessment else None,
                semantic_score=assessment.semantic_quality() if assessment else None,
                completeness_score=assessment.completeness if assessment else None,
                llm_risk_flags=list(assessment.risk_flags) if assessment else [],
            )
        )

    metadata = {
        "schema_version": "1.0",
        "video_id": workspace.video_id,
        "aspect_ratio": aspect_ratio,
        "source_video_path": str(workspace.source_video_path),
        "transcript_path": str(workspace.transcript_path),
        "highlights": [item.model_dump(mode="json") for item in rendered],
        "boundary_adjustments": [
            item.model_dump(mode="json") for item in boundary_adjustments or []
        ],
        "pipeline": pipeline_metadata or {},
    }
    (workspace_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return rendered
