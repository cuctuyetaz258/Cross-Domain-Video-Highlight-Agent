"""Các lỗi riêng của Backend"""


class MediaProcessingError(RuntimeError):
    """Lỗi khi xử lý media hoặc chạy model"""


class InvalidVideoInputError(ValueError):
    """Lỗi khi URL hoặc file local không hợp lệ"""
