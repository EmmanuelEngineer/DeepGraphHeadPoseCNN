import math
import pickle

import cv2
from mediapipe.python.solutions.drawing_utils import DrawingSpec, RED_COLOR, _BGR_CHANNELS, \
    _normalized_to_pixel_coordinates, WHITE_COLOR

from config import Config
import networkx as nx
import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np


def draw_edges(image, landmark_list, edges):
    image_rows, image_cols, _ = image.shape
    landmark_2d_coordinates = []
    for idx, landmark in enumerate(landmark_list):
        landmark_2d_coordinates.append(_normalized_to_pixel_coordinates(landmark["x"], landmark["y"],
                                                                        image_cols, image_rows))

    for edge in edges:
        start_idx = edge[0]
        end_idx = edge[1]
        cv2.line(image, landmark_2d_coordinates[start_idx],
                 landmark_2d_coordinates[end_idx], Config.ImageVisualizer.edge_color,
                 Config.ImageVisualizer.edge_thickness)


def draw_landmarks(image, landmark_list, original):
    image_rows, image_cols, _ = image.shape
    landmark_2d_coordinates = []
    for idx, landmark in enumerate(landmark_list):
        landmark_2d_coordinates.append(_normalized_to_pixel_coordinates(landmark["x"], landmark["y"],
                                                                        image_cols, image_rows))
    indexes = []  # get the original landmarx index before the extraction

    if original:
        for x in range(0, 468):
            if x not in Config.Extraction.LandmarksToIgnore.total_landmarks:
                indexes.append(x)
    else:
        indexes = range(0, len(landmark_list))

    for idx, point in enumerate(landmark_2d_coordinates):
        cv2.circle(image,
                   point,
                   Config.ImageVisualizer.landmark_thickness,
                   Config.ImageVisualizer.landmark_color, -1)
        cv2.putText(image, str(indexes[idx]), point, cv2.FONT_HERSHEY_SIMPLEX,
                    Config.ImageVisualizer.landmark_text_height,
                    Config.ImageVisualizer.landmark_color,
                    1)


def data_saver(location, data_to_save):
    try:
        with open(location, "wb") as d:
            pickle.dump(data_to_save, d)
    except Exception as ex:
        print("Cannot dump oracle:", ex)


def data_loader(path):
    try:
        with open(path, "rb") as df:
            data = pickle.load(df)
    except Exception as ex:
        print("Cannot load data" + ex)
    return data
