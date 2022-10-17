import sys

import cv2
import networkx as nx
import pandas as pd
import stellargraph
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from stellargraph import StellarGraph
from stellargraph.layer import GraphConvolution, SortPooling
from stellargraph.mapper import PaddedGraphGenerator

import Config
import GraphGenerator
import ImageAnalizer
import DataUtils
import numpy as np

from RegressorGenerator import networkx_list_to_pandas_list

images_paths = ImageAnalizer.import_images_paths(Config.image_dataset)
point_array = ImageAnalizer.landmark_extraction(images_paths)


for index, image_path in enumerate(images_paths):
    oracle_list = ImageAnalizer.extract_oracle([image_path])
    image = cv2.imread(image_path)
    image = image[:, :, ::-1].copy()
    edge_list = GraphGenerator.edge_list_generator(point_array[0])
    image_rows, image_cols, _ = image.shape
    if Config.Extraction.scale_landmarks:
        image_rows = 1000
        image_cols = 1000
    image = np.zeros((image_rows, image_cols, 3), dtype="uint8")
    DataUtils.draw_edges(image,
                         point_array[index],
                         [(x[0], x[1]) for x in edge_list])
    DataUtils.draw_landmarks(image, point_array[index], original=True)
    graph = nx.Graph()
    nodes_to_graph = []
    for idn, x in enumerate(point_array[0]):
        nodes_to_graph.append((idn, x))  # Because networkx needs the node index to work
    graph.add_nodes_from(nodes_to_graph)
    graph.add_edges_from(edge_list)
    fig = plt.figure(figsize=[10, 10], dpi=300)
    plt.title("Resultant Image")
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    custom_objects = {"GraphConvolution": GraphConvolution, "SortPooling": SortPooling}
    try:
        model = load_model(Config.working_directory + "model.h5", custom_objects=custom_objects)
    except Exception as ex:
        print("No model Found")
        print(ex)
        sys.exit(0)

    pandas_graph_list = networkx_list_to_pandas_list([graph])
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)
    stellargraph_graphs = []
    for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
        stellargraph_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
    test_generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
    obj = test_generator.flow(stellargraph_graphs)
    predict = model.predict_generator(obj)

    print(predict)





