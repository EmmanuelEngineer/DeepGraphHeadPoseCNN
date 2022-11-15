import math
import pickle

import cv2
import pandas as pd
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from sklearn.preprocessing import MinMaxScaler
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


def plot_history_scaled(history, scaler,  individual_figsize=(7, 4), return_figure=False, **kwargs):
    """
    Plot the training history of one or more models.

    This creates a column of plots, with one plot for each metric recorded during training, with the
    plot showing the metric vs. epoch. If multiple models have been trained (that is, a list of
    histories is passed in), each metric plot includes multiple train and validation series.

    Validation data is optional (it is detected by metrics with names starting with ``val_``).

    Args:
        history: the training history, as returned by :meth:`tf.keras.Model.fit`
        individual_figsize (tuple of numbers): the size of the plot for each metric
        return_figure (bool): if True, then the figure object with the plots is returned, None otherwise.
        kwargs: additional arguments to pass to :meth:`matplotlib.pyplot.subplots`

    Returns:
        :class:`matplotlib.figure.Figure`: The figure object with the plots if ``return_figure=True``, None otherwise

    """

    # explicit colours are needed if there's multiple train or multiple validation series, because
    # each train series should have the same color. This uses the global matplotlib defaults that
    # would be used for a single train and validation series.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_train = colors[0]
    color_validation = colors[1]

    if not isinstance(history, list):
        history = [history]

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix):]

    metrics = sorted({remove_prefix(m, "val_") for m in history[0].history.keys()})

    width, height = individual_figsize
    overall_figsize = (width, len(metrics) * height)

    # plot each metric in a column, so that epochs are aligned (squeeze=False, so we don't have to
    # special case len(metrics) == 1 in the zip)
    fig, all_axes = plt.subplots(
        len(metrics), 1, squeeze=False, sharex="col", figsize=overall_figsize, **kwargs
    )

    has_validation = False
    for ax, m in zip(all_axes[:, 0], metrics):
        for h in history:
            # summarize history for metric m
            ax.plot(scaler.inverse_transform(h.history[m]), c=color_train)

            try:
                val = scaler.inverse_transform(h.history["val_" + m])
            except KeyError:
                # no validation data for this metric
                pass
            else:
                ax.plot(val, c=color_validation)
                has_validation = True

        ax.set_ylabel(m, fontsize="x-large")

    # don't be redundant: only include legend on the top plot
    labels = ["train"]
    if has_validation:
        labels.append("validation")
    all_axes[0, 0].legend(labels, loc="best", fontsize="x-large")

    # ... and only label "epoch" on the bottom
    all_axes[-1, 0].set_xlabel("epoch", fontsize="x-large")

    # minimise whitespace
    fig.tight_layout()

    if return_figure:
        return fig
