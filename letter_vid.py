import os
import cv2
import mediapipe as mp

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to identify sign language letter based on landmarks
def detect_sign_language_letter(landmarks):
    letter = None

    # "A" (thumb closed, other fingers open)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "A"
    
    # "B" (palm open, other fingers straight)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "B"

    # "C" (fingers in a curved position)
    if landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "C"

    # "D" (index finger raised, other fingers closed)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "D"
    
    # "E" (fist with thumb out)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "E"
    
    # "F" (thumb and index touching, other fingers closed)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "F"

    # "G" (thumb up, index and middle out)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "G"
    
    # "H" (index and middle up, other fingers folded)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "H"
    
    # "I" (little finger up, other fingers folded)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "I"
    
    # "J" (little finger making a "J" shape)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "J"
    
    # "K" (index and middle fingers up, other fingers folded)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "K"
    
    # "L" (thumb and index finger forming "L" shape)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "L"
    
    # "M" (thumb closed over index, middle, and ring fingers)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "M"
    
    # "N" (thumb over index and middle fingers)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "N"
    
    # "O" (fist with thumb and index touching)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "O"
    
    # "P" (index and middle up, other fingers folded)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "P"
    
    # "Q" (thumb and little finger up)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "Q"
    
    # "R" (index and middle fingers up, others folded)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "R"
    
    # "S" (fist with thumb over fingers)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "S"
    
    # "T" (thumb up, other fingers folded)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "T"
    
    # "U" (index and middle fingers up, others folded)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "U"
    
    # "V" (index and middle fingers up, others folded)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "V"
    
    # "W" (index, middle, and ring fingers up)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "W"
    
    # "X" (fist with index pointing out)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "X"
    
    # "Y" (thumb and little finger up)
    if landmarks[4].x < landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y > landmarks[18].y:
        letter = "Y"
    
    # "Z" (index and middle fingers up, others folded, drawing Z shape)
    if landmarks[4].x > landmarks[3].x and landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y and landmarks[16].y > landmarks[14].y and landmarks[20].y < landmarks[18].y:
        letter = "Z"
    
    # Return detected letter
    return letter

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

                # Detect sign language letter
                letter = detect_sign_language_letter(landmarks)

                # Display detected letter or if no letter detected
                if letter:
                    cv2.putText(frame, f'Letter: {letter}', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, 'Letter: Unknown', (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Show the frame
        cv2.imshow("Sign Language Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
