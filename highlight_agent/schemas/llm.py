"""Schema cho tầng đánh giá ngữ nghĩa bằng LLM."""

from __future__ import annotations

import math
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

LLMRiskFlag = Literal[
    "cuts_sentence",
    "needs_prior_context",
    "incomplete_ending",
    "weak_hook",
    "possible_hallucination",
    "sensitive_content",
]


class CandidateTranscriptContext(BaseModel):
    """Transcript cục bộ quanh một candidate, không chứa toàn bộ video."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    ltr_score: float
    before: str
    core: str = Field(min_length=1)
    after: str

    @model_validator(mode="after")
    def validate_context(self) -> Self:
        if self.end_time <= self.start_time:
            raise ValueError("candidate context end must be greater than start")
        if not math.isfinite(self.ltr_score):
            raise ValueError("candidate context LTR score must be finite")
        return self


class LLMHighlightAssessment(BaseModel):
    """Đánh giá có cấu trúc do LLM trả về cho đúng một candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    candidate_id: str = Field(min_length=1)
    semantic_relevance: float = Field(
        ge=0,
        le=1,
        description="Mức quan trọng và ý nghĩa của nội dung cốt lõi, từ 0 đến 1.",
    )
    standalone_value: float = Field(
        ge=0,
        le=1,
        description="Khả năng hiểu clip mà không cần xem phần còn lại, từ 0 đến 1.",
    )
    completeness: float = Field(
        ge=0,
        le=1,
        description="Mức trọn vẹn về mở đầu, diễn giải và kết thúc, từ 0 đến 1.",
    )
    hook_strength: float = Field(
        ge=0,
        le=1,
        description="Khả năng phần mở đầu giữ sự chú ý, từ 0 đến 1.",
    )
    shareability: float = Field(
        ge=0,
        le=1,
        description="Giá trị khiến người xem muốn chia sẻ clip, từ 0 đến 1.",
    )
    title: str = Field(
        min_length=1,
        max_length=120,
        description="Tiêu đề ngắn, cụ thể và được transcript hỗ trợ.",
    )
    summary: str = Field(
        min_length=1,
        max_length=600,
        description="Tóm tắt vì sao đoạn này đáng xem, không thêm dữ kiện ngoài transcript.",
    )
    evidence: str = Field(
        min_length=1,
        max_length=500,
        description="Trích đoạn ngắn hoặc diễn giải sát transcript làm bằng chứng.",
    )
    suggested_start_time: float | None = Field(
        description="Timestamp có thật trong context để mở đầu trọn câu, hoặc null.",
    )
    suggested_end_time: float | None = Field(
        description="Timestamp có thật trong context để kết thúc trọn câu, hoặc null.",
    )
    risk_flags: list[LLMRiskFlag] = Field(
        max_length=6,
        description="Các rủi ro quan sát được; dùng danh sách rỗng nếu không có.",
    )

    @model_validator(mode="after")
    def validate_boundary_pair(self) -> Self:
        has_start = self.suggested_start_time is not None
        has_end = self.suggested_end_time is not None
        if has_start != has_end:
            raise ValueError("LLM boundary suggestion must contain both start and end or neither")
        if has_start:
            assert self.suggested_start_time is not None
            assert self.suggested_end_time is not None
            if not math.isfinite(self.suggested_start_time) or not math.isfinite(
                self.suggested_end_time
            ):
                raise ValueError("LLM boundary suggestions must be finite")
            if self.suggested_end_time <= self.suggested_start_time:
                raise ValueError("suggested end must be greater than suggested start")
        return self

    def semantic_quality(self) -> float:
        """Điểm semantic tổng hợp dùng cho bootstrap reranking."""

        return float(
            0.30 * self.semantic_relevance
            + 0.20 * self.standalone_value
            + 0.25 * self.completeness
            + 0.10 * self.hook_strength
            + 0.15 * self.shareability
        )


class LLMHighlightAssessmentBatch(BaseModel):
    """Structured output của một lần gọi LLM."""

    model_config = ConfigDict(extra="forbid")

    assessments: list[LLMHighlightAssessment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_candidate_ids(self) -> Self:
        ids = [item.candidate_id for item in self.assessments]
        if len(ids) != len(set(ids)):
            raise ValueError("LLM assessment candidate IDs must be unique")
        return self


class LLMRunInfo(BaseModel):
    """Metadata vận hành, không chứa API key hoặc raw transcript."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    enabled: bool
    applied: bool
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    cache_hit: bool = False
    assessed_count: int = Field(default=0, ge=0)
    fallback_reason: str | None = None
    accepted_boundary_candidate_ids: list[str] = Field(default_factory=list)
