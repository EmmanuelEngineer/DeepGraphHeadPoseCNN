import math
import pickle
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import DataUtils
from config import Config
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import math
import matplotlib.pyplot as plt
import numpy as np
import time

data = None
oracle = None
try:
    with open(Config.dataset_folder + "data_array.pickle", "rb") as df:
        data = pickle.load(df)
except Exception as ex:
    print("Cannot load data" + ex)

try:
    with open(Config.dataset_folder + "oracle_list.pickle", "rb") as of:
        oracle = pickle.load(of)
except Exception as ex:
    print("Cannot load data" + ex)


def node_distance(a, b):
    d = math.sqrt(math.pow(b["x"] - a["x"], 2) +
                  math.pow(b["y"] - a["y"], 2) +
                  math.pow(b["z"] - a["z"], 2) * 1.0)
    return d


def get_max_from_mins(array):  # get max edge
    maximum = [0, 0]
    for idx, x in enumerate(array):
        if x[1] > maximum[1]:
            maximum = x
    maximum_distance = maximum[1]
    index_of_maximum = maximum[0]
    return array, maximum_distance, index_of_maximum


def substitute_max_in_array(array, index, index_of_maximum, distance):
    for x in array:
        if x[0] == index_of_maximum:
            array.remove(x)
            break
    array.append((index, distance))
    get_max_from_mins(array)
    return array


def edge_list_generator(node_list):
    # For each node links the five closest nodes with an edge and the distance as weight
    edge_list = []
    for a_index, a_node in enumerate(node_list):
        n_of_temp = 5
        array_of_mins = []
        for b_index, b_node in enumerate(node_list):
            if (a_index == b_index) or ((b_index, a_index) in [(n[0], n[1]) for n in edge_list]): # check for same node or edge already created
                continue
            if n_of_temp > 0:
                array_of_mins.append((b_index, node_distance(a_node, b_node)))
                n_of_temp -= 1
                continue
            array_of_mins, maximum_distance, index_of_maximum = get_max_from_mins(array_of_mins)
            distance = node_distance(a_node, b_node)
            if distance < maximum_distance:
                array_of_mins= substitute_max_in_array(array_of_mins, b_index, index_of_maximum, distance)

        for node in array_of_mins:
            edge_list.append((a_index, node[0], {"weight": node[1]}))

    return edge_list


def generate_graph(element):  # elements will contain the subject number and the list of nodes
    number = element[0]
    nodes = element[1]
    nodes_to_graph = []
    graph = nx.Graph()
    edges = edge_list_generator(nodes)
    for idn, x in enumerate(nodes):
        nodes_to_graph.append((idn, x))  # Because networkx needs the edge index to work
    graph.add_nodes_from(nodes_to_graph)
    graph.add_edges_from(edges)
    return number, graph


# Executive Code
begin = time.time()
array_of_graphs = []
with ProcessPoolExecutor(max_workers=Config.n_of_threads) as exe:
    iterable_of_graphs = exe.map(generate_graph, enumerate(data))
    for subject_number, i in iterable_of_graphs:
        print(f"{subject_number} on {len(data) - 1} complete.")
        array_of_graphs.append(i)
end = time.time()
print("Completed in ", end - begin)
print(array_of_graphs)


def data_saver(location, oracle_to_dump):
    try:
        with open(location, "wb") as d:
            pickle.dump(oracle_to_dump, d)
    except Exception as ex:
        print("Cannot dump oracle:", ex)


data_saver(Config.dataset_folder + "array_graph_networkx.pickle", array_of_graphs)

"""
image = np.zeros((500, 500, 3), dtype = "uint8")
DataUtils.draw_landmarks(image,
                         data[1],
                         landmark_drawing_spec=DrawingSpec())
plt.title("Resultant Image");
plt.axis('off');
plt.imshow(image);
plt.show()"""
