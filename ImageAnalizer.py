import os
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

    array_of_directories = [Config.image_dataset + name + "/*" for name in os.listdir(Config.image_dataset)
                            if os.path.isdir(os.path.join(Config.image_dataset, name))]

    print(array_of_directories)
    array_of_glob = []
    for x in array_of_directories:
        array_of_glob.append(glob.glob(x))

    return array_of_glob


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
        oracle_array.append({"pitch": float(arr[1]), "yaw": float(arr[3]), "roll": float(arr[5])})
        counter += 1
    print("total counted = ", counter)
    return oracle_array


def scale_landmarks(list_of_landmarks):
    dataframe = pd.DataFrame.from_records(list_of_landmarks)
    scaler = MinMaxScaler()
    scaler.fit(dataframe)
    d = scaler.transform(dataframe)
    dataframe = pd.DataFrame(d, columns=dataframe.columns)
    return dataframe.to_dict("records")


def landmark_extraction(list_of_paths):
    extracted_landmarks = []
    no_face_at = []
    #   setting up the extractor
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_images = mp_face_mesh.FaceMesh(
        static_image_mode=Config.Extraction.static_image_mode,
        max_num_faces=Config.Extraction.max_num_faces)
    print("number of considered landmark per face: %d" % (
            468 - len(Config.Extraction.LandmarksToIgnore.total_landmarks)))
    for id_im, _image_path in enumerate(list_of_paths):
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
        else:
            print("No face found at:", _image_path)
            no_face_at.append(id_im)
            continue

        if Config.Extraction.scale_landmarks:
            filtered_landmarks = scale_landmarks(filtered_landmarks)
        extracted_landmarks.append(filtered_landmarks)
    return extracted_landmarks, no_face_at


if __name__ == "__main__":
    images_paths_all_subjects = import_images_paths(Config.image_dataset)
    oracle_all_subjects = extract_oracle(images_paths_all_subjects)
    id_to_exclude = []
    landmark_array_all_subjects = []
    for subject_id, subject_images in enumerate(images_paths_all_subjects):
        landmark_array_of_subjects, id_to_exclude_of_subjects = landmark_extraction(subject_images)
        print("Total faces non found: ", len(id_to_exclude))
        oracle = [ele for idx, ele in enumerate(oracle_all_subjects[subject_id]) if idx not in id_to_exclude]
        landmark_array_all_subjects.append(landmark_array_of_subjects)

    DataUtils.data_saver(Config.working_directory+"landmarks_by_subject.pickle", landmark_array_all_subjects)
    DataUtils.data_saver(Config.working_directory+"oracle_by_subject.pickle", landmark_array_all_subjects)

    for x in landmark_array_all_subjects:
        print(len(landmark_array_all_subjects))


