import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import Config
import DataUtils


#   This class is designed to extract the dataset from a series of images
#   and output an array of landmarks with the relative information
#   stored with a dictionary.
#   The oracle is stored as a second array with a dictionary


def import_images_paths(dataset_directory):
    import glob

    list_of_paths = glob.glob(dataset_directory)
    return list_of_paths


def extract_oracle(paths_of_images):
    import re

    # the oracle of each image is saved in the file name as pitch, yaw, roll
    # this function will read the file name return the oracle as an array
    regex = "(frame_\\d\\d\\d\\d\\d_)(([+]|-).*)(([+]|-).*)(([+]|-).*)[.]png"
    oracle_array = []
    counter = 0
    for index, single_oracle in enumerate(paths_of_images):
        list_of_matches = re.findall(regex, single_oracle)
        arr = list_of_matches.pop()
        oracle_array.append({"index": index, "pitch": float(arr[1]), "yaw": float(arr[3]), "roll": float(arr[5])})
        counter += 1
    print("total counted = ", counter)
    return oracle_array


def scale_landmarks(list_of_landmarks):
    dataframe = pd.DataFrame.from_records(list_of_landmarks)
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    d = scaler.transform(numpy.array(dataframe))
    dataframe = pd.DataFrame(d, columns=dataframe.columns)
    return dataframe.to_dict("records")


def landmark_extraction(list_of_paths):
    extracted_landmarks = []
    #   setting up the extractor
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(
        static_image_mode=Config.Extraction.static_image_mode,
        max_num_faces=Config.Extraction.max_num_faces)
    for _image_path in list_of_paths:
        image = cv2.imread(_image_path)

        # Extraction
        face_mesh_results = face_mesh_images.process(image[:, :, ::-1])
        image = None

        if face_mesh_results.multi_face_landmarks:  # Check if face is recognised

            filtered_landmarks = []  # Exclusion of mobile parts of the face
            for face_landmarks in face_mesh_results.multi_face_landmarks:  # Should activate only once for one face
                for idl, land in enumerate(face_landmarks.landmark):
                    if idl not in Config.Extraction.LandmarksToIgnore.total_landmarks:
                        # Filtering landmarks
                        filtered_landmarks.append({"x": land.x, "y": land.y,
                                                   "z": land.z})  # Translating mediapipe object in a generic array

        if Config.Extraction.scale_landmarks:
            filtered_landmarks = scale_landmarks(filtered_landmarks)
        extracted_landmarks.append(filtered_landmarks)
    return extracted_landmarks

"""
images_paths = import_images_paths(Config.image_dataset)
oracle = extract_oracle(images_paths)
DataUtils.data_saver(Config.dataset_folder + "oracle_list.pickle", oracle)

landmark_array = landmark_extraction(images_paths)
print(type(landmark_array), "length", len(landmark_array))
print(oracle)
DataUtils.data_saver(Config.dataset_folder + "data_array.pickle", landmark_array)
"""