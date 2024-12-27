import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture history for smoothing predictions
gesture_history = deque(maxlen=5)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Function to recognize hand gesture
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    thumb_folded = thumb_tip[0] < thumb_ip[0]

    # Gesture logic
    if thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "A"
    elif all(finger[1] > wrist[1] for finger in [index_tip, middle_tip, ring_tip, pinky_tip]) and not thumb_folded:
        return "B"
    elif thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip]) and all(finger[1] > wrist[1] for finger in [ring_tip, pinky_tip]):
        return "C"
    elif thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] > wrist[1]:
        return "D"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "E"
    elif not thumb_folded and index_tip[1] < wrist[1] and all(finger[1] > wrist[1] for finger in [middle_tip, ring_tip, pinky_tip]):
        return "F"
    elif thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and all(finger[1] > wrist[1] for finger in [ring_tip, pinky_tip]):
        return "G"
    elif thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip]) and all(finger[1] > wrist[1] for finger in [ring_tip, pinky_tip]):
        return "H"
    elif thumb_folded and index_tip[1] > wrist[1] and middle_tip[1] < wrist[1]:
        return "I"
    elif thumb_folded and index_tip[1] > wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] < wrist[1]:
        return "J"
    elif thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip, ring_tip]) and pinky_tip[1] > wrist[1]:
        return "K"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip, ring_tip]) and pinky_tip[1] < wrist[1]:
        return "L"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "M"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip]) and ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1]:
        return "N"
    elif thumb_folded and index_tip[1] > wrist[1] and middle_tip[1] > wrist[1] and ring_tip[1] > wrist[1] and pinky_tip[1] > wrist[1]:
        return "O"
    elif thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip]) and ring_tip[1] > wrist[1] and pinky_tip[1] > wrist[1]:
        return "P"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip]) and ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1]:
        return "Q"
    elif thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip]) and all(finger[1] > wrist[1] for finger in [ring_tip, pinky_tip]):
        return "R"
    elif thumb_folded and index_tip[1] < wrist[1] and all(finger[1] > wrist[1] for finger in [middle_tip, ring_tip, pinky_tip]):
        return "S"
    elif thumb_folded and all(finger[1] > wrist[1] for finger in [index_tip, middle_tip]) and ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1]:
        return "T"
    elif not thumb_folded and all(finger[1] < wrist[1] for finger in [index_tip, middle_tip]):
        return "U"
    elif not thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] > wrist[1]:
        return "V"
    elif not thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] < wrist[1] and pinky_tip[1] > wrist[1]:
        return "W"
    elif thumb_folded and index_tip[1] > wrist[1] and middle_tip[1] < wrist[1]:
        return "X"
    elif thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] > wrist[1] and ring_tip[1] > wrist[1] and pinky_tip[1] < wrist[1]:
        return "Y"
    elif thumb_folded and index_tip[1] < wrist[1] and middle_tip[1] > wrist[1] and ring_tip[1] > wrist[1] and pinky_tip[1] > wrist[1]:
        return "Z"

    return "Unknown"

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Convert landmarks to a list of tuples
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                # Recognize gesture
                gesture = recognize_gesture(landmarks)
                gesture_history.append(gesture)

                # Most common gesture in history
                most_common = max(set(gesture_history), key=gesture_history.count)

                # Display gesture
                cv2.putText(frame, f"Gesture: {most_common}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("ASL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
