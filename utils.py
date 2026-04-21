import cv2
import mediapipe as mp


def get_face_landmarks(image, draw=False, static_image_mode=True):
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=1,
        min_detection_confidence=0.5
    )

    results = face_mesh.process(image_input_rgb)

    image_landmarks = []

    if results.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        single_face = results.multi_face_landmarks[0].landmark

        xs_ = []
        ys_ = []
        zs_ = []

        for landmark in single_face:
            xs_.append(landmark.x)
            ys_.append(landmark.y)
            zs_.append(landmark.z)

        for i in range(len(xs_)):
            image_landmarks.append(xs_[i] - min(xs_))
            image_landmarks.append(ys_[i] - min(ys_))
            image_landmarks.append(zs_[i] - min(zs_))

    face_mesh.close()
    return image_landmarks