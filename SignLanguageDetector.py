import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Iterate over each hand
        for hand_landmarks in results.multi_hand_landmarks:
            # You can use hand_landmarks to get information about hand landmarks
            # Add your sign language detection logic here

            # For example, you can print the coordinates of the thumb tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            print(f"Thumb Tip Coordinates: {thumb_tip.x}, {thumb_tip.y}, {thumb_tip.z}")

            # Draw hand landmarks on the frame
            # Move this line inside the for loop
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop when 'x' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release the capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()