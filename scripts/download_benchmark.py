"""Script tải nhanh bộ dataset TVSum và SumMe (.h5) từ Google Drive Mirror."""

import zipfile
from pathlib import Path

import gdown


def main():
    benchmark_dir = Path("data/benchmark")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Google Drive File ID cho bộ dataset video summarization (chứa TVSum, SumMe .h5)
    file_id = "11ulsvk1MZI7iDqymw9cfL7csAYS0cDYH"
    zip_path = benchmark_dir / "datasets.zip"

    print("Dang tai TVSum & SumMe datasets tu Google Drive...")
    gdown.download(id=file_id, output=str(zip_path), quiet=False)

    print("\nDang giai nen cac file .h5...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Chỉ giải nén các file .h5
            if member.filename.endswith(".h5"):
                filename = Path(member.filename).name
                target_path = benchmark_dir / filename
                with zf.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())
                size_mb = target_path.stat().st_size / (1024 * 1024)
                print(f" -> Da giai nen: {filename} ({size_mb:.2f} MB)")

    # Xóa file zip tạm
    if zip_path.exists():
        zip_path.unlink()
        print("Da xoa file datasets.zip tam.")

    print("\nHOAN THANH! Danh sach file benchmark san sang:")
    for h5_file in benchmark_dir.glob("*.h5"):
        size_mb = h5_file.stat().st_size / (1024 * 1024)
        print(f" * {h5_file.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
