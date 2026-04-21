import os
import cv2
import numpy as np
from utils import get_face_landmarks

data_dir = r'C:\Users\gaado\OneDrive\المستندات\مجلد جديد\data\train'

def read_image_unicode(path):
    image_data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    return image

output = []

emotions = sorted([
    folder for folder in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, folder))
    and folder.lower() in ['angry', 'happy', 'sad']
])

print("Detected classes:", emotions)

for emotion_index, emotion in enumerate(emotions):
    emotion_path = os.path.join(data_dir, emotion)
    print("Processing folder:", emotion)

    for image_name in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_name)
        print("Reading image:", image_path)

        image = read_image_unicode(image_path)

        if image is None:
            print("Could not read image:", image_path)
            continue

        face_landmarks = get_face_landmarks(image)

        print("Landmarks length:", len(face_landmarks))

        if len(face_landmarks) == 1404:
            face_landmarks.append(emotion_index)
            output.append(face_landmarks)
        else:
            print("Skipped image:", image_name)

print("Total samples extracted:", len(output))

if len(output) > 0:
    np.savetxt("data.txt", np.asarray(output))
    print("Data saved successfully to data.txt")
else:
    print("No data extracted")