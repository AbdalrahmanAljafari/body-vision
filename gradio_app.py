import json
from uuid import uuid4

import cv2
import gradio as gr

from app.services.standardization_service import StandardizationService
from app.services.pose_service import PoseService


standardization_service = StandardizationService()
pose_service = PoseService()


def read_image_rgb(image_path: str):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def run_pipeline(front_image_path: str, side_image_path: str, subject_height_cm: float):
    if not front_image_path:
        raise gr.Error("Please upload front image.")
    if not side_image_path:
        raise gr.Error("Please upload side image.")
    if subject_height_cm is None or subject_height_cm <= 0:
        raise gr.Error("Height must be greater than 0.")

    with open(front_image_path, "rb") as f:
        front_bytes = f.read()

    with open(side_image_path, "rb") as f:
        side_bytes = f.read()

    job_id = str(uuid4())

    # 1) Standardization
    result = standardization_service.standardize_pair(
        front_bytes=front_bytes,
        side_bytes=side_bytes,
        subject_height_cm=subject_height_cm,
        job_id=job_id,
    )

    front_path = result["front_standardized_path"]
    side_path = result["side_standardized_path"]

    # 2) Pose extraction
    front_landmarks = pose_service.extract_pose_landmarks(front_path)
    side_landmarks = pose_service.extract_pose_landmarks(side_path)

    # 3) Landmark previews
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

    # 4) Build JSON
    output_json = {
        "height_cm": subject_height_cm,
        "front_image_size": [2000, 2000],
        "side_image_size": [2000, 2000],
        "front_landmarks": front_landmarks,
        "side_landmarks": side_landmarks,
    }

    json_text = json.dumps(output_json, indent=2, ensure_ascii=False)

    # 5) Read images for display in Gradio
    front_standardized_img = read_image_rgb(front_path)
    side_standardized_img = read_image_rgb(side_path)
    front_preview_img = read_image_rgb(front_preview_path)
    side_preview_img = read_image_rgb(side_preview_path)

    return (
        front_standardized_img,
        side_standardized_img,
        front_preview_img,
        side_preview_img,
        json_text,
    )


with gr.Blocks(title="Body Vision Pipeline") as demo:
    gr.Markdown("# Body Vision Pipeline")
    gr.Markdown(
        "Upload front and side images, enter height, then view standardized images, "
        "landmark previews, and final JSON coordinates."
    )

    with gr.Row():
        front_input = gr.Image(label="Front Image", type="filepath")
        side_input = gr.Image(label="Side Image", type="filepath")

    height_input = gr.Number(label="Height (cm)", value=170)

    run_button = gr.Button("Run Pipeline")

    gr.Markdown("## Standardized Images")
    with gr.Row():
        front_standardized_output = gr.Image(label="Front Standardized")
        side_standardized_output = gr.Image(label="Side Standardized")

    gr.Markdown("## Landmark Preview Images")
    with gr.Row():
        front_preview_output = gr.Image(label="Front Landmarks Preview")
        side_preview_output = gr.Image(label="Side Landmarks Preview")

    gr.Markdown("## JSON Coordinates")
    json_output = gr.Code(label="Output JSON", language="json")

    run_button.click(
        fn=run_pipeline,
        inputs=[front_input, side_input, height_input],
        outputs=[
            front_standardized_output,
            side_standardized_output,
            front_preview_output,
            side_preview_output,
            json_output,
        ],
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9001))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )