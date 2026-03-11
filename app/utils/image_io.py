import cv2
import numpy as np


class InvalidImageError(Exception):
    pass


def bytes_to_rgb(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr is None:
        raise InvalidImageError("Invalid image file or unsupported format.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb