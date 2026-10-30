import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pygame, time

cap = cv2.VideoCapture(0)

pygame.mixer.init()

start_time = time.time()

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,                      # two hands -> counts up to 10
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    # Convert each landmark to pixel coordinates
    
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    # Draw lines between connected landmarks (green)
    
    for connection in mp_hands_connections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (0, 255, 0), 2)
    
    # Draw a dot at each landmark (red, filled)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0,0,255), -1)

sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    2: "sounds_swaras/2_Re.mp3",
    3: "sounds_swaras/3_Ga.mp3",
    4: "sounds_swaras/4_Ma.mp3",
    5: "sounds_swaras/5_Pa.mp3",
    6: "sounds_swaras/6_Dha.mp3",
    7: "sounds_swaras/7_Ni.mp3",
    8: "sounds_swaras/8_Sa_upper.mp3",
}
previous_count = None

while True:
    success, frame = cap.read()
    
    if not success:
        print("ERROR: Could not read frame from webcam")
        break

    mirrored_frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    count = 0
    if result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[i][0].category_name           # Labels handedness "Left" or "Right"

            # COUNTING FINGERS THAT ARE UP:
            if hand_landmarks[8].y < hand_landmarks[6].y:
                count += 1
            if hand_landmarks[12].y < hand_landmarks[10].y:
                count += 1
            if hand_landmarks[16].y < hand_landmarks[14].y:
                count += 1
            if hand_landmarks[20].y < hand_landmarks[18].y:
                count += 1

            # COUNTING IF THUMB IS UP:
            if label == "Right":
                if hand_landmarks[4].x > hand_landmarks[3].x:
                    count += 1
            else:
                if hand_landmarks[4].x < hand_landmarks[3].x:
                    count += 1
            
            draw_landmarks(mirrored_frame, hand_landmarks)

    if count != previous_count:
        if count in sounds_mapping:
            pygame.mixer.music.load(sounds_mapping[count])
            pygame.mixer.music.play()
        previous_count = count
    
    print(f"Hands: {len(result.hand_landmarks)} Fingers: {count}")

    cv2.imshow("Mirrored webcam", mirrored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()