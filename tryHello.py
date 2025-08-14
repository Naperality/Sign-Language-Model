import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import json
from collections import deque

# ----------------------
# SETTINGS
# ----------------------
MODEL_PATH = "asl_holistic_model.h5"        # path to your trained model
LABEL_ENCODER_PATH = "label_encoder.joblib" # path to your label encoder
SEQUENCE_LENGTH = 30                        # must match training
CAMERA_INDEX = 0                            # 0 for built-in webcam

# ----------------------
# LOAD MODEL & LABELS
# ----------------------
model = tf.keras.models.load_model(MODEL_PATH)
le = joblib.load(LABEL_ENCODER_PATH)
print("✅ Model and label encoder loaded")

# ----------------------
# MEDIAPIPE SETUP
# ----------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    # Pose: 33 landmarks * (x,y,z,visibility) = 132
    pose = np.zeros(33 * 4)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks]).flatten()

    # Face: only first 468 landmarks = 1404
    face = np.zeros(468 * 3)
    if results.face_landmarks:
        face_landmarks = results.face_landmarks.landmark[:468]
        face = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks]).flatten()

    # Left hand: 63
    left_hand = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    # Right hand: 63
    right_hand = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, face, left_hand, right_hand])

# ----------------------
# REAL-TIME PREDICTION
# ----------------------
sequence = deque(maxlen=SEQUENCE_LENGTH)

cap = cv2.VideoCapture(CAMERA_INDEX)
with mp_holistic.Holistic(static_image_mode=False,
                          model_complexity=1,
                          refine_face_landmarks=True,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract keypoints and add to sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Predict when sequence is full
        if len(sequence) == SEQUENCE_LENGTH:
            seq_array = np.expand_dims(sequence, axis=0)  # shape: (1, 30, 1662)
            probs = model.predict(seq_array, verbose=0)[0]
            pred_idx = np.argmax(probs)
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]

            # Display prediction
            cv2.putText(frame, f"{pred_label} ({confidence*100:.1f}%)",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("ASL Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
