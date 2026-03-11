from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.health import HealthResponse
from app.services.standardization_service import StandardizationService
from app.services.pose_service import PoseService
from app.utils.image_io import InvalidImageError


router = APIRouter(prefix="/api/v1", tags=["standardization"])

service = StandardizationService()
pose_service = PoseService()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="API is running")


@router.post("/standardize")
async def standardize(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
    subject_height_cm: float = Form(...),
):

    if subject_height_cm <= 0:
        raise HTTPException(
            status_code=400,
            detail="subject_height_cm must be greater than 0"
        )

    if not front_image.content_type or not front_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="front_image must be an image"
        )

    if not side_image.content_type or not side_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="side_image must be an image"
        )

    front_bytes = await front_image.read()
    side_bytes = await side_image.read()

    job_id = str(uuid4())

    try:
        # 1️⃣ standardization
        result = service.standardize_pair(
            front_bytes=front_bytes,
            side_bytes=side_bytes,
            subject_height_cm=subject_height_cm,
            job_id=job_id,
        )

        front_path = result["front_standardized_path"]
        side_path = result["side_standardized_path"]

        # 2️⃣ pose extraction
        front_landmarks = pose_service.extract_pose_landmarks(front_path)
        side_landmarks = pose_service.extract_pose_landmarks(side_path)

        # 3️⃣ landmark preview images
        front_preview_path = front_path.replace(
            "front_standardized.png",
            "front_landmarks_preview.png",
        )

        side_preview_path = side_path.replace(
            "side_standardized.png",
            "side_landmarks_preview.png",
        )

        pose_service.draw_landmarks_preview(
            front_path,
            front_landmarks,
            front_preview_path,
        )

        pose_service.draw_landmarks_preview(
            side_path,
            side_landmarks,
            side_preview_path,
        )

    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        ) from e

    # 4️⃣ final JSON output
    return {
        "height_cm": subject_height_cm,
        "front_image_size": [2000, 2000],
        "side_image_size": [2000, 2000],

        "front_standardized_path": front_path,
        "side_standardized_path": side_path,

        "front_landmarks": front_landmarks,
        "side_landmarks": side_landmarks,

        "front_landmarks_preview_path": front_preview_path,
        "side_landmarks_preview_path": side_preview_path,
    }