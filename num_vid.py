import os
import cv2
import mediapipe as mp

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to count raised fingers
def count_fingers(landmarks):
    fingers = []

    # Thumb: Compare tip with IP joint (x-axis)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)  # Thumb up
    else:
        fingers.append(0)

    # 4 Fingers: Compare fingertip (y) with PIP joint (y)
    fingertip_indices = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    pip_indices = [6, 10, 14, 18]

    for tip, pip in zip(fingertip_indices, pip_indices):
        if landmarks[tip].y < landmarks[pip].y:  # Tip is above PIP
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

# Start video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmarks = hand_landmarks.landmark

                # Count fingers
                finger_count = count_fingers(landmarks)

                # Display finger count on the screen
                cv2.putText(frame, f'Fingers: {finger_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        # Show the frame
        cv2.imshow("Finger Counting", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
