import math
import pickle

import cv2
import pandas as pd
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.preprocessing import MinMaxScaler
from stellargraph import StellarGraph

import Config
import networkx as nx
import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


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
    print("saving: ", location)
    try:
        with open(location, "wb") as d:
            pickle.dump(data_to_save, d)
    except Exception as ex:
        print("Cannot dump oracle:", ex)


def data_loader(path):
    print("loading: ", path)
    try:
        with open(path, "rb") as df:
            data = pickle.load(df)
    except Exception as ex:
        print("Cannot load data", ex)
    return data


def networkx_list_to_pandas_list(input_list):  # Step taken for efficiency of as of the stellargraph docs
    output_list = []
    for idx, x in enumerate(input_list):
        edges = nx.to_pandas_edgelist(x)
        nodes = pd.DataFrame.from_dict(dict(x.nodes(data=True)), orient='index')
        output_list.append({"nodes": nodes, "edges": edges})
    return output_list


def apply_RicciCurvature_on_list(networkx_list):
    output_list = []
    counter = 0
    for graph in networkx_list:
        counter += 1
        print(counter, "on", len(networkx_list))
        orf = OllivierRicci(graph, alpha=0.5, base=1, exp_power=0, proc=1, verbose="INFO")
        orf.compute_ricci_flow(iterations=10)

        for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
            for att in ["weight", "original_RC"]:
                d.pop(att, None)

        for n1, d in graph.nodes(data=True):
            d.pop("ricciCurvature", None)
        output_list.append(orf.G.copy())

    return output_list


def graph_cleaner(networkx_list, ricci_graphs=Config.RegressionSetting.apply_RicciCurvature):

    if ricci_graphs:
        for graph in networkx_list:
            for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
                for att in ["weight", "original_RC"]:
                    d.pop(att, None)

            for n1, d in graph.nodes(data=True):
                for att in ["x", "y", "z"]:
                    d.pop(att, None)
    else:
        for graph in networkx_list:
            for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
                for att in ["ricciCurvature", "original_RC"]:
                    d.pop(att, None)

            for n1, d in graph.nodes(data=True):
                for att in ["ricciCurvature"]:
                    d.pop(att, None)
    return networkx_list


def convert_pandas_graph_list_to_stellargraph(nx_list):
    sg_graphs = []
    for idx, graph_nx in enumerate(nx_list):  # Conversion to stellargraph Graphs
        sg_graphs.append(StellarGraph(
            {"landmark": graph_nx["nodes"]}, {"line": graph_nx["edges"]}))
    return sg_graphs
