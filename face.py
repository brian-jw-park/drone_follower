import cv2
import dlib
import numpy as np
from collections import deque


# Load dlib face detector and predictor for landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize buffers for smoothing
buffer_length = 10
x_buffer = deque(maxlen=buffer_length)
y_buffer = deque(maxlen=buffer_length)
angle_buffer = deque(maxlen=buffer_length)

# Initialize previous position
prev_x, prev_y = None, None

def moving_average(buffer):
    return sum(buffer) / len(buffer) if buffer else 0

def get_face_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    all_face = [
        (face.left(), face.top(), face.width(), face.height()) for face in faces
    ]

    return all_face


def get_face_data(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Get the landmarks
        landmarks = predictor(gray, face)
        landmarks_points = [(p.x, p.y) for p in landmarks.parts()]

        # Calculate center of the face
        center_x = (x + x + w) // 2
        center_y = (y + y + h) // 2

        # Track movement
        if prev_x is not None and prev_y is not None:
            movement_x = center_x - prev_x
            movement_y = center_y - prev_y

            # Update buffers
            x_buffer.append(movement_x)
            y_buffer.append(movement_y)

            # Calculate smoothed movement
            avg_movement_x = moving_average(x_buffer)
            avg_movement_y = moving_average(y_buffer)

            cv2.putText(img, f"Move X: {avg_movement_x:.2f}, Move Y: {avg_movement_y:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Update previous position
        prev_x, prev_y = center_x, center_y

        # Calculate angular movement
        left_eye = np.array([landmarks_points[36], landmarks_points[39]])  # Left eye landmarks
        right_eye = np.array([landmarks_points[42], landmarks_points[45]]) # Right eye landmarks

        left_eye_center = left_eye.mean(axis=0).astype("int")
        right_eye_center = right_eye.mean(axis=0).astype("int")

        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # Update angle buffer
        angle_buffer.append(angle)

        # Calculate smoothed angle
        avg_angle = moving_average(angle_buffer)

        cv2.putText(img, f"Angle: {avg_angle:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw rectangle around the face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw landmarks
        for point in landmarks_points:
            cv2.circle(img, point, 2, (0, 0, 255), -1)

    return img
