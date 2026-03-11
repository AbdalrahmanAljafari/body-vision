from pydantic import BaseModel


class StandardizationResponse(BaseModel):
    job_id: str
    subject_height_cm: float
    canvas_size: list[int]
    front_standardized_path: str
    side_standardized_path: str
    front_body_height_px: float
    side_body_height_px: float