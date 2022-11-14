import sys

import cv2
import networkx as nx
import pandas as pd
from keras.saving.save import load_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from stellargraph import StellarGraph
from stellargraph.layer import GraphConvolution, SortPooling
from stellargraph.mapper import PaddedGraphGenerator

import Config
import GraphGenerator
import ImageAnalizer
import DataUtils
import numpy as np

images_paths = ImageAnalizer.import_images_paths(Config.Test.image_dataset)
point_array, x = ImageAnalizer.landmark_extraction(images_paths)

custom_objects = {"GraphConvolution": GraphConvolution, "SortPooling": SortPooling}
try:
    model = load_model(Config.working_directory + "model.h5", custom_objects=custom_objects)
except Exception as ex:
    print("No model Found")
    print(ex)
    sys.exit(0)

print(images_paths)

model.summary()

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
    if True:
        graph = DataUtils.apply_RicciCurvature_on_list([graph])[0]

        for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
            for att in ["weight", "original_RC"]:
                d.pop(att, None)

        for n1, d in graph.nodes(data=True):
            for att in ["x", "y", "z"]:
                d.pop(att, None)

    print(graph.nodes(data=True))
    print(graph.edges(data=True))
    fig = plt.figure(figsize=[10, 10], dpi=300)
    plt.title("Resultant Image")
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    pandas_graph_list = DataUtils.networkx_list_to_pandas_list([graph])
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)
    result_scaler = MinMaxScaler().fit(pandas_oracle)

    pandas_graph_list[0]["nodes"].to_csv("nodes.csv")
    pandas_graph_list[0]["edges"].to_csv("edges.csv")

    stellargraph_graphs = []
    for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
        stellargraph_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
    test_generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
    obj = test_generator.flow(stellargraph_graphs)
    predict = model.predict(obj)
    print("no Scaling", predict)
    predict = result_scaler.inverse_transform(predict)
    print("predicted", predict, "expected", oracle_list[0])
