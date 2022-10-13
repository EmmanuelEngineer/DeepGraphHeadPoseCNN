import cv2
import networkx as nx
from matplotlib import pyplot as plt

import Config
import GraphGenerator
import ImageAnalizer
import DataUtils
import numpy as np

images_paths = ImageAnalizer.import_images_paths(Config.image_dataset)
point_array = ImageAnalizer.landmark_extraction(images_paths)

for index, image_path in enumerate(images_paths):
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

# generator = PaddedGraphGenerator(graphs=graphs)
