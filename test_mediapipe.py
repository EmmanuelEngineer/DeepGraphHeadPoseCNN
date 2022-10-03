import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
from config import Config

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=Config.Extraction.static_image_mode,
                                         max_num_faces=Config.Extraction.max_num_faces)

mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles
stringa = Config.image_dataset + '1/frame_00003_+007.61+003.29-001.57.png'
print(stringa)
sample_img = cv2.imread(stringa)
plt.figure(figsize=[10, 10])

plt.title("Sample Image")
plt.axis('off')
plt.imshow(sample_img[:, :, ::-1])
plt.show()

face_mesh_results = face_mesh_images.process(sample_img[:, :, ::-1])  # Analisi Immagine dopo conversione da RGB a BGR

LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))  # Liste di occhi sinistri
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))    # Liste di occhi Destri

if face_mesh_results.multi_face_landmarks:

    for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):

        print(f'FACE NUMBER: {face_no + 1}')
        print('-----------------------')

        print(f'LEFT EYE LANDMARKS:n')

        for LEFT_EYE_INDEX in LEFT_EYE_INDEXES[:2]:
            print(face_landmarks.landmark[LEFT_EYE_INDEX])
            # Recupera il punto corrispondente all'occhio passando
            # l'indice

        print(f'RIGHT EYE LANDMARKS:n')

        for RIGHT_EYE_INDEX in RIGHT_EYE_INDEXES[:2]:
            print(face_landmarks.landmark[RIGHT_EYE_INDEX])

img_copy = sample_img[:, :, ::-1].copy()  # Copia l'immagine

if face_mesh_results.multi_face_landmarks:

    for face_landmarks in face_mesh_results.multi_face_landmarks:
        mp_drawing.draw_landmarks(image=img_copy,
                                  landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        mp_drawing.draw_landmarks(image=img_copy, landmark_list=face_landmarks,
                                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

fig = plt.figure(figsize=[10, 10])
plt.title("Resultant Image");
plt.axis('off');
plt.imshow(img_copy);
plt.show()
