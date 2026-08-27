"""Module trích xuất highlight bằng LLM — Prompt Engineering bởi Thảo Anh"""

import json
import os

from openai import OpenAI

from highlight_agent.schemas.highlight import HighlightCandidate

# ================================================================
# HỆ THỐNG PROMPT ĐA MIỀN (CROSS-DOMAIN PROMPT SYSTEM)
# ================================================================

# --- PROMPT CHÍNH (dùng chung cho mọi miền) ---
BASE_SYSTEM_PROMPT = """Bạn là một chuyên gia sản xuất video ngắn (Shorts/Reels/TikTok) hàng đầu thế giới.

NHIỆM VỤ:
Phân tích transcript (bản ghi lời nói có kèm mốc thời gian tính bằng giây) của một video dài, sau đó chọn ra chính xác {highlight_count} khoảnh khắc đáng xem nhất để cắt thành clip ngắn viral.

⚠️ QUY TẮC QUAN TRỌNG NHẤT (CRITICAL — VI PHẠM SẼ BỊ TỪ CHỐI):
Thời lượng mỗi clip BẮT BUỘC phải từ 30 đến 90 giây: 30.0 <= (end_time - start_time) <= 90.0.
- Nếu đoạn hay chỉ dài 10–20 giây, hãy MỞ RỘNG bao gồm thêm phần setup (mở đầu, dẫn dắt) phía trước hoặc phần payoff (kết luận, phản ứng) phía sau để đạt tối thiểu 30 giây.
- Thời lượng lý tưởng nhất là 45–60 giây (vừa đủ hấp dẫn, không quá dài).
- TUYỆT ĐỐI KHÔNG chọn clip dưới 30 giây dù đoạn đó hay đến mấy.

VÍ DỤ KIỂM TRA THỜI LƯỢNG:
- start=120.0, end=165.0 → duration=45.0s ✓ HỢP LỆ (lý tưởng)
- start=200.0, end=260.0 → duration=60.0s ✓ HỢP LỆ
- start=80.0, end=95.0 → duration=15.0s ✗ QUÁ NGẮN → phải lùi start hoặc kéo dài end thêm ít nhất 15s nữa
- start=50.0, end=180.0 → duration=130.0s ✗ QUÁ DÀI → phải thu hẹp, chỉ giữ phần hay nhất

QUY TẮC BẮT BUỘC KHÁC:
1. Các clip KHÔNG ĐƯỢC chồng lấn thời gian lên nhau.
2. Các clip phải phân bố đều trong video (không chọn các đoạn dính liền sát nhau).
3. start_time và end_time PHẢI nằm trong phạm vi thời gian của transcript (không bịa số).
4. Điểm cắt (start/end) nên trùng với ranh giới câu nói tự nhiên — không cắt giữa chừng một câu.
5. Chấm điểm (score) từ 0.0 đến 10.0 dựa trên mức độ hấp dẫn và khả năng viral.
6. Sắp xếp kết quả theo score giảm dần (clip hay nhất lên đầu).

{domain_criteria}

CÁCH LÀM VIỆC — SUY NGHĨ TỪNG BƯỚC TRƯỚC KHI CHỌN:
Bước 1: Đọc toàn bộ transcript, xác định loại video và tổng thời lượng.
Bước 2: Đánh dấu 5–7 đoạn tiềm năng có nội dung hấp dẫn.
Bước 3: Với mỗi đoạn, xác định start_time và end_time sao cho bao trọn ý (từ câu mở đầu setup đến câu kết thúc payoff).
Bước 4: TÍNH duration = end_time - start_time. Nếu < 30s, MỞ RỘNG thêm ngữ cảnh xung quanh. Nếu > 90s, THU HẸP giữ phần hay nhất.
Bước 5: Kiểm tra các clip có chồng lấn không, có phân bố đều không.
Bước 6: Chọn đúng {highlight_count} clip tốt nhất, chấm điểm và sắp xếp.

TIÊU CHÍ CHẤM ĐIỂM CHUNG:
- Có "hook" mạnh ngay đầu clip (câu mở đầu gây tò mò, gây sốc, hoặc phản trực giác) → +2 điểm
- Có kết thúc tròn trịa (không bỏ lửng ý) → +1 điểm
- Có tính độc lập (người xem hiểu được mà không cần xem toàn bộ video) → +1 điểm
- Có giá trị chia sẻ (người xem muốn gửi cho bạn bè) → +2 điểm
- Có yếu tố cảm xúc mạnh (bất ngờ, hài hước, cảm động, tức giận) → +2 điểm

✅ KIỂM TRA LẦN CUỐI trước khi xuất JSON:
□ Mỗi clip có duration nằm trong [30, 90] giây? (Tính lại: end_time - start_time)
□ Các clip có chồng lấn thời gian lên nhau không?
□ Điểm cắt start/end có trùng ranh giới câu nói tự nhiên không?
□ Có đúng {highlight_count} clip không?
□ start_time và end_time có nằm trong phạm vi transcript không?

ĐỊNH DẠNG ĐẦU RA — BẮT BUỘC JSON THUẦN, KHÔNG CÓ BẤT KỲ CHỮ NÀO KHÁC:
{{
  "highlights": [
    {{
      "candidate_id": "hl_01",
      "start_time": 120.5,
      "end_time": 175.0,
      "score": 9.2,
      "reason": "Giải thích chi tiết vì sao đoạn này đáng xem, những tín hiệu nào khiến bạn chọn nó",
      "signals": {{
        "hook_strength": 0.9,
        "emotional_peak": 0.8,
        "shareability": 0.85
      }}
    }}
  ]
}}

LƯU Ý VỀ TRƯỜNG "signals":
- Mỗi signal là một con số từ 0.0 đến 1.0, đại diện cho mức độ mạnh của tín hiệu đó.
- Bạn PHẢI đưa ra ít nhất 2 signals cho mỗi highlight.
- Tên signal phải bằng tiếng Anh, dùng snake_case.
"""

# --- TIÊU CHÍ RIÊNG THEO MIỀN ---
DOMAIN_CRITERIA = {
    "lecture": """TIÊU CHÍ RIÊNG CHO BÀI GIẢNG (LECTURE):
Đây là video bài giảng/giáo dục. Hãy ưu tiên chọn:
- Khoảnh khắc "Eureka/Aha": khi giảng viên giải thích xong một khái niệm khó bằng ví dụ cực kỳ trực quan, dễ hiểu.
- Sự thật gây sốc hoặc phản trực giác: thông tin khiến người học phải dừng lại suy nghĩ.
- Ứng dụng thực tế: khi lý thuyết được liên hệ với đời sống hàng ngày.
- Tóm tắt cô đọng: đoạn giảng viên tóm lại toàn bộ ý chính trong 1-2 câu.
KHÔNG nên chọn: phần giới thiệu chung chung, phần đọc slide, phần chào hỏi/kết thúc.
Signals gợi ý: concept_clarity, surprise_factor, practical_value, information_density.""",

    "podcast": """TIÊU CHÍ RIÊNG CHO PODCAST/TỌA ĐÀM:
Đây là video podcast hoặc tọa đàm có nhiều người nói. Hãy ưu tiên chọn:
- Tranh luận nảy lửa: hai người bất đồng quan điểm mạnh mẽ, có lập luận sắc bén.
- Quote đáng nhớ: câu nói "đóng đinh" vào đầu người nghe, có thể trích dẫn độc lập.
- Khoảnh khắc cảm xúc: khi khách mời chia sẻ câu chuyện cá nhân sâu sắc hoặc bất ngờ.
- Góc nhìn phản biện: khi ai đó lật ngược hoàn toàn một quan điểm phổ biến.
- Tiết lộ bất ngờ: thông tin mới lần đầu được công bố hoặc bí mật được hé lộ.
KHÔNG nên chọn: phần giới thiệu khách mời, small talk, phần quảng cáo/sponsor.
Signals gợi ý: debate_intensity, quote_memorability, emotional_depth, perspective_shift.""",

    "standup": """TIÊU CHÍ RIÊNG CHO HÀI ĐỘC THOẠI (STAND-UP COMEDY):
Đây là video hài độc thoại. Hãy ưu tiên chọn:
- Punchline cực mạnh: câu đùa có setup dài và punchline bất ngờ, gây cười sảng khoái.
- Bit hoàn chỉnh: một đoạn hài có mở đầu, phát triển, và kết thúc tròn trịa (không cắt giữa bit).
- Callback: khi comedian nhắc lại một câu đùa trước đó theo cách bất ngờ.
- Tương tác với khán giả: khoảnh khắc comedian ứng biến với phản ứng của khán giả.
- Đoạn leo thang (escalation): khi câu đùa được đẩy lên nhiều tầng, mỗi tầng hài hơn tầng trước.
KHÔNG nên chọn: phần chào hỏi, phần chuyển tiếp giữa các bit, phần nói lan man không có punchline.
LƯU Ý ĐẶC BIỆT: Transcript hài thường thiếu ngữ cảnh (tiếng cười, giọng điệu). Hãy dựa vào CẤU TRÚC câu đùa (setup → pause → punchline) để nhận diện.
Signals gợi ý: punchline_impact, bit_completeness, escalation_level, audience_reaction."""
}

# Prompt mặc định nếu không xác định miền
DOMAIN_CRITERIA["auto"] = """TIÊU CHÍ CHUNG (CHƯA XÁC ĐỊNH MIỀN):
Hãy tự phân tích nội dung transcript để xác định loại video, sau đó áp dụng tiêu chí phù hợp:
- Nếu là bài giảng: ưu tiên kiến thức, ví dụ trực quan, sự thật bất ngờ.
- Nếu là podcast: ưu tiên tranh luận, quote, cảm xúc.
- Nếu là hài: ưu tiên punchline, bit hoàn chỉnh, tiếng cười.
Signals gợi ý: tùy thuộc vào loại video bạn nhận diện được."""


def build_prompt(domain: str = "auto", highlight_count: int = 3) -> str:
    """Ghép prompt chính + tiêu chí riêng theo miền."""
    criteria = DOMAIN_CRITERIA.get(domain, DOMAIN_CRITERIA["auto"])
    return BASE_SYSTEM_PROMPT.format(
        highlight_count=highlight_count,
        domain_criteria=criteria,
    )


def _fix_duration(item: dict) -> dict:
    """
    Tự động chỉnh sửa start_time/end_time để duration nằm trong [30, 90] giây.
    - Nếu clip quá ngắn (< 30s): kéo dài end_time cho đủ 30s.
    - Nếu clip quá dài (> 90s): cắt ngắn end_time cho còn 90s.
    """
    start = float(item.get("start_time", 0))
    end = float(item.get("end_time", 0))
    duration = end - start

    if duration < 30:
        # Clip quá ngắn → kéo dài end_time
        item["end_time"] = round(start + 45.0, 1)  # Đặt mặc định 45s (giữa 30-90)
        print(f"  → Sửa {item.get('candidate_id')}: quá ngắn ({duration:.1f}s) → kéo dài thành 45s")
    elif duration > 90:
        # Clip quá dài → cắt ngắn end_time
        item["end_time"] = round(start + 60.0, 1)  # Đặt mặc định 60s
        print(f"  → Sửa {item.get('candidate_id')}: quá dài ({duration:.1f}s) → cắt còn 60s")

    return item


def extract_highlights_from_transcript(
    transcript_text: str,
    domain: str = "auto",
    highlight_count: int = 3,
) -> list[HighlightCandidate]:
    """Gửi transcript cho Groq AI và lấy về danh sách Highlight chuẩn Pydantic."""

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("Chưa có GROQ_API_KEY trong file .env!")

    client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

    system_prompt = build_prompt(domain, highlight_count)

    # Giới hạn transcript gửi đi (tối ưu hóa cho Groq Free Tier: TPM <= 8000)
    # 17500 ký tự ~ 4500 input tokens + max_tokens 1200 = ~5700 tokens (an toàn dưới 8000 TPM)
    max_chars = 17500
    if len(transcript_text) <= max_chars:
        short_transcript = transcript_text
    else:
        truncated = transcript_text[:max_chars]
        # Tìm dấu chấm câu cuối cùng để cắt đúng ranh giới câu
        last_period = max(truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! "), truncated.rfind(".\n"))
        if last_period > max_chars * 0.8:
            short_transcript = truncated[:last_period + 1]
        else:
            short_transcript = truncated

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=1200,
        temperature=0.6,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n{short_transcript}"},
        ],
    )

    # Parse JSON từ response (robust: xử lý markdown, text thừa)
    raw_text = (response.choices[0].message.content or "").strip()
    # Loại bỏ markdown ```json ... ```
    if "```" in raw_text:
        import re
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw_text, re.DOTALL)
        if code_match:
            raw_text = code_match.group(1).strip()
    # Fallback: tìm JSON object {...} đầu tiên
    if not raw_text.startswith("{"):
        start_idx = raw_text.find("{")
        if start_idx != -1:
            raw_text = raw_text[start_idx:]
    raw_json = json.loads(raw_text)

    # In ra JSON thô để debug
    print("\n========== AI TRẢ VỀ (RAW JSON) ==========")
    for i, item in enumerate(raw_json.get("highlights", [])):
        start = item.get("start_time", "?")
        end = item.get("end_time", "?")
        try:
            dur = float(end) - float(start)
            status = "✓ OK" if 30 <= dur <= 90 else f"✗ FAIL ({dur:.1f}s)"
        except (ValueError, TypeError):
            dur = "?"
            status = "✗ FAIL (không parse được)"
        print(f"  [{i+1}] {item.get('candidate_id')}: {start}s → {end}s | duration={dur} | {status}")
    print("=" * 45 + "\n")

    candidates = []
    for item in raw_json.get("highlights", []):
        # Tự động sửa duration nếu AI tính sai
        item = _fix_duration(item)
        try:
            candidate = HighlightCandidate(**item)
            candidates.append(candidate)
        except Exception as e:
            print(f"Cảnh báo: Bỏ qua candidate không hợp lệ {item.get('candidate_id', 'unknown')}: {e}")

    # Sắp xếp theo điểm giảm dần
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
