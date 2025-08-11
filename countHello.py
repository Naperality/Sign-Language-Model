import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===== Load model & labels =====
MODEL_PATH = "asl_lstm_model.h5"  # your trained model
ACTIONS = ['hello', 'nothing' , 'thank_you']  # must match training exactly!

model = tf.keras.models.load_model(MODEL_PATH)

# ===== MediaPipe setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ===== Function to extract landmarks from frame =====
def extract_landmarks_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand = [0] * 63
    right_hand = [0] * 63

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if handedness.classification[0].label == "Left":
                left_hand = landmarks
            else:
                right_hand = landmarks

    return left_hand + right_hand  # 126 values

# ===== Webcam Capture =====
cap = cv2.VideoCapture(0)
sequence = []
SEQUENCE_LENGTH = 30  # must match training
last_label = ""
prev_action = None
action_counts = {action: 0 for action in ACTIONS}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get landmarks for current frame
    landmarks = extract_landmarks_from_frame(frame)
    sequence.append(landmarks)
    sequence = sequence[-SEQUENCE_LENGTH:]  # keep only last N frames

    # Predict when we have enough frames
    if len(sequence) == SEQUENCE_LENGTH:
        seq_padded = pad_sequences([sequence], maxlen=SEQUENCE_LENGTH, padding='post', dtype='float32')
        pred = model.predict(seq_padded, verbose=0)
        pred_class = np.argmax(pred)
        confidence = np.max(pred)
        current_action = ACTIONS[pred_class]

        last_label = f"{current_action} ({confidence:.2f})"

        # Count when confidence is high enough and action detected
        if confidence > 0.7:
            if prev_action is None:  # No ongoing action
                action_counts[current_action] += 1
                prev_action = current_action
        else:
            prev_action = None  # Reset when confidence drops


    # Display last prediction
    cv2.putText(frame, last_label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Display counts
    y_offset = 80
    for action, count in action_counts.items():
        cv2.putText(frame, f"{action}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        y_offset += 30

    # Draw landmarks for visualization
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
