import io
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.segmentor import get_segmentor
from app.utils.image_io import bytes_to_rgb


class StandardizationService:
    def __init__(self):
        self.output_root = Path(settings.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _segment_person_and_bbox(self, rgb_image: np.ndarray, thresh: float = 0.2):
        h, w, _ = rgb_image.shape

        segmentor = get_segmentor()
        results = segmentor.process(rgb_image)

        if results.segmentation_mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(
                mask,
                (int(0.2 * w), int(0.1 * h)),
                (int(0.8 * w), int(0.95 * h)),
                255,
                -1,
            )
        else:
            mask = (results.segmentation_mask > thresh).astype(np.uint8) * 255

        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            min_x, max_x = 0, w - 1
            min_y, max_y = 0, h - 1
        else:
            min_x, max_x = int(xs.min()), int(xs.max())
            min_y, max_y = int(ys.min()), int(ys.max())

        return mask, (min_x, min_y, max_x, max_y)

    def _standardize_single_view(
        self,
        image_bytes: bytes,
        subject_height_cm: float,
        job_id: str,
        view_name: str,
    ) -> str:
        rgb = bytes_to_rgb(image_bytes)
        orig_h, orig_w, _ = rgb.shape

        mask, (min_x, min_y, max_x, max_y) = self._segment_person_and_bbox(rgb)
        body_h_px = max(max_y - min_y, 1)

        target_body_px = settings.canvas_height_px * (subject_height_cm / settings.canvas_height_cm)
        scale = target_body_px / float(body_h_px)

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(resized_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            min_x_r, max_x_r = 0, new_w - 1
            min_y_r, max_y_r = 0, new_h - 1
        else:
            min_x_r, max_x_r = int(xs.min()), int(xs.max())
            min_y_r, max_y_r = int(ys.min()), int(ys.max())

        canvas = np.ones(
            (settings.canvas_height_px, settings.canvas_width_px, 3),
            dtype=np.uint8,
        ) * 255

        body_center_x = (min_x_r + max_x_r) / 2.0
        canvas_center_x = settings.canvas_width_px / 2.0
        shift_x = int(canvas_center_x - body_center_x)

        target_feet_y = settings.canvas_height_px - 1
        shift_y = int(target_feet_y - max_y_r)

        x1_src, y1_src = 0, 0
        x2_src, y2_src = new_w, new_h

        x1_dst = shift_x
        y1_dst = shift_y
        x2_dst = shift_x + new_w
        y2_dst = shift_y + new_h

        if x1_dst < 0:
            x1_src -= x1_dst
            x1_dst = 0
        if y1_dst < 0:
            y1_src -= y1_dst
            y1_dst = 0
        if x2_dst > settings.canvas_width_px:
            x2_src -= x2_dst - settings.canvas_width_px
            x2_dst = settings.canvas_width_px
        if y2_dst > settings.canvas_height_px:
            y2_src -= y2_dst - settings.canvas_height_px
            y2_dst = settings.canvas_height_px

        x1_src = max(0, x1_src)
        y1_src = max(0, y1_src)
        x2_src = min(new_w, x2_src)
        y2_src = min(new_h, y2_src)

        if x1_src < x2_src and y1_src < y2_src:
            person_region = resized_rgb[y1_src:y2_src, x1_src:x2_src]
            canvas[y1_dst:y2_dst, x1_dst:x2_dst] = person_region

        out_dir = self.output_root / job_id
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{view_name}_standardized.png"
        Image.fromarray(canvas).save(out_path)

        return str(out_path)

    def standardize_pair(
        self,
        front_bytes: bytes,
        side_bytes: bytes,
        subject_height_cm: float,
        job_id: str,
    ) -> dict:
        front_path = self._standardize_single_view(
            image_bytes=front_bytes,
            subject_height_cm=subject_height_cm,
            job_id=job_id,
            view_name="front",
        )

        side_path = self._standardize_single_view(
            image_bytes=side_bytes,
            subject_height_cm=subject_height_cm,
            job_id=job_id,
            view_name="side",
        )

        return {
    "front_standardized_path": front_path,
    "side_standardized_path": side_path,
    }