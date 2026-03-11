import os
import cv2
import mediapipe as mp


class PoseService:
    LANDMARK_NAMES = {
        "nose": 0,
        "left_eye_inner": 1,
        "left_eye": 2,
        "left_eye_outer": 3,
        "right_eye_inner": 4,
        "right_eye": 5,
        "right_eye_outer": 6,
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)

    def extract_pose_landmarks(self, image_path: str) -> dict:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            raise ValueError(f"No pose landmarks detected for image: {image_path}")

        h, w, _ = image_bgr.shape
        landmarks = results.pose_landmarks.landmark

        output = {}
        for name, idx in self.LANDMARK_NAMES.items():
            lm = landmarks[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)
            output[name] = [x, y]

        return output

    def draw_landmarks_preview(self, image_path: str, landmarks_dict: dict, output_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        for name, (x, y) in landmarks_dict.items():
            # Draw the point
            cv2.circle(image, (x, y), 8, (0, 0, 255), -1)

            # Write the name of the point
            cv2.putText(
                image,
                name,
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

        return output_path