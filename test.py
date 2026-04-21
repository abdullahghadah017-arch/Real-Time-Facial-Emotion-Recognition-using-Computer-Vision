import pickle
import cv2
from utils import get_face_landmarks

with open("model.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
emotions = saved_data["emotions"]

print("Loaded classes:", emotions)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame from webcam")
        break

    face_landmarks = get_face_landmarks(
        frame,
        draw=True,
        static_image_mode=False
    )

    if len(face_landmarks) == 1404:
        output = model.predict([face_landmarks])
        predicted_emotion = emotions[int(output[0])]

        cv2.putText(
            frame,
            predicted_emotion,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("Emotion Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()