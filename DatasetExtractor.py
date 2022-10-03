import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from config import Config
import DataUtils


#   This class is designed to extract the dataset from a series of images
#   and output an array of landmarks with the relative information
#   stored with a dictionary.
#   The oracle is stored as a second array with a dictionary


class Data_Extractor:
    @staticmethod
    def import_images_paths(dataset_directory):
        import glob

        list_of_paths = glob.glob(dataset_directory)
        return list_of_paths

    @staticmethod
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

    @staticmethod
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
            extracted_landmarks.append(filtered_landmarks)
        return extracted_landmarks

"""
images_paths = Data_Extractor.import_images_paths(Config.image_dataset)
oracle = Data_Extractor.extract_oracle(images_paths)
DataUtils.data_saver(Config.dataset_folder + "oracle_list.pickle", oracle)

landmark_array = Data_Extractor.landmark_extraction(images_paths)
print(type(landmark_array), "length", len(landmark_array))
DataUtils.data_saver(Config.dataset_folder + "data_array.pickle", landmark_array)"""