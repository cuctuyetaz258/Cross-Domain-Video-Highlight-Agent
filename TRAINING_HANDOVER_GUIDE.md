# HƯỚNG DẪN BÀN GIAO TIẾN ĐỘ & QUY TRÌNH HUẤN LUYỆN (TRAINING HANDOVER GUIDE)

> **Mục đích tài liệu:** Cung cấp đầy đủ ngữ cảnh kiến trúc, tiến độ triển khai thực tế theo `TRAINING_PLAN.md` và hướng dẫn từng bước (kèm câu lệnh chi tiết) để tiếp tục huấn luyện trên máy mới hoặc nạp context cho AI coding assistant.

---

## 1. Tổng quan kiến trúc hệ thống

Dự án **Cross-Domain Video Highlight Agent** xây dựng hệ thống phát hiện highlight video tối ưu cho 2 domain chính: `lecture` (bài giảng) và `podcast` (đối thoại).

### 1.1 LTR Scorer (Learning-to-Rank)
- **7-Channel Feature Matrix ($7 \times T$, sample rate 2.0 Hz):**
  1. `rms`: Cường độ năng lượng âm thanh (root-mean-square).
  2. `pitch`: Tần số cơ bản giọng nói ($F_0$ qua pyin).
  3. `silence`: Khoảng lặng âm thanh (silence ratio).
  4. `text_score`: TF-IDF keyword density từ transcript.
  5. `scene_change`: Tín hiệu phát hiện chuyển cảnh visual (PySceneDetect ContentDetector).
  6. `gesture`: Mật độ chuyển động cử chỉ/tương tác visual (RAFT Optical Flow / Frame Differencing).
  7. `turn_rate`: Tần suất đổi lượt nói (Speaker Diarization / Turn transition).
- **Contract chuẩn:** Window 5.0s (10 mẫu), hop 1.0s (2 mẫu), tham chiếu độ dài $L_{\text{ref}} = 68.0$, schema version 1.1.
- **Mô hình:** `AdditiveAttentionScorer` sử dụng channel projection + additive attention layer + margin ranking hinge loss ($\gamma = 1.0$) + temporal smoothness penalty ($\lambda_{\text{smooth}} = 0.01$).

### 1.2 LLM Scoring (Semantic Rerank)
- Bỏ cơ chế 5 trọng số heuristic cũ. LLM chỉ trả về một điểm số duy nhất:
  $$\text{llm\_score} = \text{overall\_quality} \in [0, 1]$$
- LLM đánh giá candidate dựa trên transcript đoạn highlight, context xung quanh và prompt định dạng structured output.

### 1.3 LTR–LLM Fusion
- **Percentile / Rank Normalization:** Với mỗi video, xếp hạng các candidate theo từng nguồn score riêng biệt và chuẩn hóa về $[0, 1]$:
  $$\text{rank\_score} = \frac{N - \text{rank}}{N - 1}$$
- **Score kết hợp:**
  $$\text{final\_score} = \alpha \cdot \text{rank\_score}_{\text{ltr}} + (1 - \alpha) \cdot \text{rank\_score}_{\text{llm}}$$
- **Tối ưu $\alpha$:** Grid search $\alpha \in [0, 1]$ (bước 0.05) trên tập validation để cực đại hóa `macro_nDCG@3`. (Đã loại bỏ biến thể Fixed Baseline 0.60/0.40 theo yêu cầu).

---

## 2. Bảng theo dõi tiến độ chi tiết (Đối chiếu TRAINING_PLAN.md)

| Giai đoạn | Nội dung | Trạng thái | Chi tiết artifact & Kết quả đạt được |
|---|---|:---:|---|
| **Phase 0** | **Chuẩn bị dữ liệu & nhãn** | 🟢 **100% Sẵn sàng** | • **10/10 custom CSV** (5 lecture, 5 podcast) đã gắn nhãn importance 1–5 theo khoảng 2s.<br>• **10/10 boundary JSON** xác định ground truth highlight.<br>• **10/10 media folder** (`data/raw/in_domain_pilot/<video_id>/`) đầy đủ `source_video.mp4`, `audio.wav`, `transcript.json`.<br>• **75/75 benchmark cache** (TVSum 50 + SumMe 25) đã trích xuất chuẩn v1.1 trong `data/features_cache/`.<br>• 5-fold manifest (`data/manifests/custom_fold{0..4}.jsonl`) đã phân chia 3 train / 1 val / 1 test mỗi domain. |
| **Phase 1** | **Pretrain LTR (TVSum + SumMe)** | 🟢 **Hoàn thành** | • Checkpoint: `data/models/ltr_pretrained_tvsum_summe.pt`<br>• Huấn luyện trên CUDA RTX 4050, Best Epoch 1, `macro_source_AP = 0.6287`, Test AP = `0.6933`.<br>• Lịch sử: `data/reports/pretraining_history.csv`, `data/reports/pretraining_curves.svg`, `data/reports/ltr_evaluation_v1_1_cuda.json`. |
| **Phase 2** | **Fine-tune LTR (Lecture & Podcast)** | 🟡 **Sẵn sàng chạy** | • Chờ trích xuất 10 custom cache v1.1 trên máy mới.<br>• Chạy 5-fold cross-validation khởi tạo từ `ltr_pretrained_tvsum_summe.pt` và huấn luyện release checkpoint `ltr_target_lecture_podcast.pt`. |
| **Phase 3** | **LLM Assessment Cache** | 🟢 **Code sẵn sàng** | • Module `highlight_agent/llm/reranker.py` đã chuẩn hóa `overall_quality`. |
| **Phase 4 & 5** | **Fusion Dataset & Grid Search $\alpha$** | 🟢 **Code sẵn sàng** | • Scripts `scripts/build_fusion_dataset.py` và `evaluation/train_fusion.py` đã hoàn thiện và vượt qua 100% unit tests (205 tests). |
| **Phase 6 & 7** | **Evaluation & Đóng gói Artifact** | 🟡 **Chờ kết quả Phase 2 & 5** | • Đóng gói 2 file độc lập: `ltr_target_lecture_podcast.pt` và `fusion_calibrator.json`. |

---

## 3. Cấu trúc thư mục quan trọng

```text
Cross-Domain-Video-Highlight-Agent/
├── data/
│   ├── annotations/
│   │   ├── raw/                       # 10 CSV nhãn 1-5 của custom lecture & podcast
│   │   └── boundaries/                # 10 file boundary JSON ground truth
│   ├── manifests/
│   │   ├── tvsum_summe.jsonl          # Manifest benchmark 75 video
│   │   ├── custom_fold0.jsonl         # 5-fold cross-validation manifests
│   │   ├── ...
│   │   └── custom_fold4.jsonl
│   ├── models/
│   │   ├── ltr_pretrained_tvsum_summe.pt      # Pretrained LTR checkpoint (Phase 1)
│   │   ├── ltr_pretrained_tvsum_summe_last.pt
│   │   └── ltr_target_lecture_podcast.pt      # Mục tiêu Phase 2 (sẽ sinh ra)
│   ├── raw/
│   │   └── in_domain_pilot/           # 10 thư mục chứa video, audio, transcript
│   │       ├── -cRswJf8OnI/ (source_video.mp4, audio.wav, transcript.json)
│   │       ├── 1bszFX_XcbU/
│   │       ├── aircAruvnKk/
│   │       ├── DNQDqq4mWSY/
│   │       ├── g2-_pnmhO4A/
│   │       ├── IHZwWFHWa-w/
│   │       ├── u36A-YTxiOw/
│   │       ├── waLjtcUq5Mc/
│   │       ├── wjZofJX0v4M/
│   │       └── WUvTyaaNkzM/
│   └── reports/                       # Logs CSV, biểu đồ SVG và report JSON
├── highlight_agent/
│   ├── features/                      # Trích xuất 7 kênh đặc trưng
│   ├── llm/                           # OpenAI LLM client, reranker & fusion logic
│   └── models/
│       ├── ltr_scorer.py              # PyTorch AdditiveAttentionScorer model
│       └── train_offline.py           # Training engine LTR
├── scripts/
│   ├── build_feature_cache.py         # Trích xuất feature cache 7 kênh
│   ├── prepare_custom_manifest.py     # Tạo 5-fold manifest
│   └── build_fusion_dataset.py        # Tạo dataset huấn luyện fusion
├── evaluation/
│   ├── train_fusion.py                # Grid search alpha calibrator
│   └── evaluate_ltr.py                # Đánh giá LTR checkpoint
├── TRAINING_PLAN.md                   # Kế hoạch chi tiết gốc
└── TRAINING_HANDOVER_GUIDE.md         # File tài liệu này
```

---

## 4. Hướng dẫn thiết lập môi trường trên máy mới

### 4.1 Cài đặt môi trường Python (Khuyến nghị Conda Python 3.11)
```bash
conda create -n MLIoT python=3.11 -y
conda activate MLIoT

# Cài đặt PyTorch với CUDA (ví dụ CUDA 11.8 hoặc 12.1 tùy GPU máy mới)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt các thư viện cần thiết
pip install opencv-python numpy scipy scikit-learn librosa soundfile pydantic pytest pytest-asyncio
```

### 4.2 Kiểm tra kiểm thử tự động
Trước khi chạy training, chạy test suite để xác nhận môi trường đã hoàn chỉnh:
```bash
pytest tests
# Kỳ vọng: 205 passed
```

---

## 5. Quy trình huấn luyện từng bước trên máy mới

### BƯỚC 1: Trích xuất 10 Feature Caches cho Custom In-Domain Video
Chạy lệnh sau để build/rebuild toàn bộ feature cache cho 10 video in-domain với Extractor v1.1:
```bash
python scripts/build_feature_cache.py \
  --manifest data/manifests/custom_fold0.jsonl \
  --output-dir data/features_cache \
  --force \
  --device cuda \
  --report data/reports/custom_cache_build.json
```
*(Nếu máy không có CUDA, đổi `--device cpu`)*.

---

### BƯỚC 2: Huấn luyện 5-Fold Cross Validation cho LTR (Phase 2)
Huấn luyện lần lượt 5 fold bằng cách fine-tune từ `ltr_pretrained_tvsum_summe.pt`:

```bash
# Fold 0
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold0.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold0.pt \
  --last-output data/models/custom_fold0_last.pt \
  --training-log data/reports/custom_fold0_log.json \
  --training-history-csv data/reports/custom_fold0_history.csv \
  --training-plot data/reports/custom_fold0_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4

# Fold 1
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold1.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold1.pt \
  --last-output data/models/custom_fold1_last.pt \
  --training-log data/reports/custom_fold1_log.json \
  --training-history-csv data/reports/custom_fold1_history.csv \
  --training-plot data/reports/custom_fold1_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4

# Fold 2
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold2.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold2.pt \
  --last-output data/models/custom_fold2_last.pt \
  --training-log data/reports/custom_fold2_log.json \
  --training-history-csv data/reports/custom_fold2_history.csv \
  --training-plot data/reports/custom_fold2_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4

# Fold 3
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold3.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold3.pt \
  --last-output data/models/custom_fold3_last.pt \
  --training-log data/reports/custom_fold3_log.json \
  --training-history-csv data/reports/custom_fold3_history.csv \
  --training-plot data/reports/custom_fold3_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4

# Fold 4
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold4.jsonl \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/custom_fold4.pt \
  --last-output data/models/custom_fold4_last.pt \
  --training-log data/reports/custom_fold4_log.json \
  --training-history-csv data/reports/custom_fold4_history.csv \
  --training-plot data/reports/custom_fold4_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4
```

---

### BƯỚC 3: Huấn luyện Release Checkpoint LTR Mục Tiêu
Sau khi có kết quả 5 fold, huấn luyện mô hình release `ltr_target_lecture_podcast.pt` trên toàn bộ tập custom training:
```bash
python -m highlight_agent.models.train_offline \
  --manifest data/manifests/custom_fold0.jsonl \
  --train-split train \
  --val-split val \
  --cache-dir data/features_cache \
  --init-checkpoint data/models/ltr_pretrained_tvsum_summe.pt \
  --source-weight custom_scores=1.0 \
  --output data/models/ltr_target_lecture_podcast.pt \
  --last-output data/models/ltr_target_lecture_podcast_last.pt \
  --training-log data/reports/ltr_target_log.json \
  --training-history-csv data/reports/ltr_target_history.csv \
  --training-plot data/reports/ltr_target_curves.svg \
  --max-epochs 50 \
  --patience 15 \
  --lr 1e-4
```

---

### BƯỚC 4: Tạo Fusion Dataset & Grid Search $\alpha$ (Phase 4 & 5)
1. Chạy sinh candidate và đánh giá LLM để tạo metadata cho từng video.
2. Chạy ghép dataset:
```bash
python scripts/build_fusion_dataset.py \
  --manifest data/manifests/custom_fold0.jsonl \
  --metadata-dir output/fusion_runs \
  --output data/manifests/custom_fusion.jsonl
```
3. Chạy tối ưu $\alpha$ và sinh `fusion_calibrator.json`:
```bash
python evaluation/train_fusion.py \
  --dataset data/manifests/custom_fusion.jsonl \
  --split val \
  --output data/models/fusion_calibrator.json \
  --report data/reports/fusion_training_report.json
```

---

### BƯỚC 5: Đánh giá Toàn diện & Nghiệm thu (Phase 6 & 7)
- Chạy script đánh giá so sánh:
  1. **LTR-only** (sử dụng `ltr_target_lecture_podcast.pt`)
  2. **LLM-only** (chỉ dùng `overall_quality`)
  3. **Learned Global $\alpha$** (Rank-normalized kết hợp qua `fusion_calibrator.json`)
- Báo cáo rõ các chỉ số theo từng domain (`lecture_nDCG`, `podcast_nDCG`) và `macro_nDCG`.
