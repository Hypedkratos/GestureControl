import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Mediapipe Hand Detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)

# Set up Pycaw for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame for easier hand movement
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            # Convert normalized coordinates to pixel values
            h, w, _ = frame.shape
            thumb_tip_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw circles on thumb and index finger tips
            cv2.circle(frame, thumb_tip_coords, 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, index_tip_coords, 10, (0, 0, 255), cv2.FILLED)
            cv2.line(frame, thumb_tip_coords, index_tip_coords, (0, 255, 0), 3)

            # Calculate distance between thumb and index finger tips
            distance = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))

            # Map the distance to volume range
            vol = np.interp(distance, [20, 200], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Display volume level on screen
            cv2.putText(frame, f'Volume: {int(np.interp(vol, [min_vol, max_vol], [0, 100]))} %',
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Gesture Volume Control', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
