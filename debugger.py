import cv2
import networkx as nx
from matplotlib import pyplot as plt
import GraphGenerator
from DatasetExtractor import Data_Extractor
import DataUtils

"""
# TODO inserire codice per importare il dataset

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
print(summary.describe().round(1))
#graph_labels = pd.get_dummies(graph_labels, drop_first=True)#Ottenere nomi generici di colonne
"""

images_paths = Data_Extractor.import_images_paths("/home/emmavico/Desktop/temp/*")
point_array = Data_Extractor.landmark_extraction(images_paths)

for index, image_path in enumerate(images_paths):
    image = cv2.imread(image_path)
    image = image[:, :, ::-1].copy()
    edge_list = GraphGenerator.edge_list_generator(point_array[0])
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
    for node in graph.edges():
        print(node)

    fig = plt.figure(figsize=[10, 10], dpi=300)
    plt.title("Resultant Image")
    plt.axis('off')
    plt.imshow(image)
    plt.show()

# generator = PaddedGraphGenerator(graphs=graphs)
