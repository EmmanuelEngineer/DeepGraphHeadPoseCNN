import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import DataUtils
import Config
import networkx as nx
import math

data = DataUtils.data_loader(Config.working_directory + "data_array.pickle")


def node_distance(a, b):
    array_1, array_2 = np.array([x for x in a.values()]), np.array([x for x in b.values()])
    squared_distance = np.sum(np.square(array_1 - array_2))
    distance = np.sqrt(squared_distance)
    return distance


def already_taken(edge, selected_edges):
    counter = 0
    flag = False
    a_index = edge[0]
    b_index = edge[1]

    for edge in selected_edges:
        if edge[0] == a_index and edge[1] == b_index:  # The opposite edge exist
            flag = False
            break

    if flag:
        return True
    else:
        return False


def sort(edge):
    return edge[2]


def edge_list_generator(node_list):
    selected_edges = []
    all_edges = [] # archive of all possible edges grouped by starting edge
    for a_index, a_node in enumerate(node_list):
        all_edges.append([])
        for b_index, b_node in enumerate(node_list):
            if a_index == b_index:
                continue

            all_edges[a_index].append((a_index, b_index,
                                       node_distance(node_list[a_index], node_list[b_index])))
        all_edges[a_index].sort(key=sort) # sorts all the possible edges from a single point by eucludian distance
    for index in range(0, len(node_list)):
        for edge in all_edges[index][:Config.Extraction.edges_per_landmark]: # selects the closest edges
            if not already_taken(edge, selected_edges): # doesn't stop the counter because skips the already taken edges
                selected_edges.append(edge)

    selected_edges1 = []
    for x in selected_edges:
        selected_edges1.append((x[0], x[1], {"weight": x[2]}))
    return selected_edges1


def generate_graph(element):  # elements will contain the subject number and the list of landmarks
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
"""

# Executive Code
begin = time.time()
array_of_graphs = []
with ProcessPoolExecutor(max_workers=Config.n_of_threads) as exe:
    iterable_of_graphs = exe.map(generate_graph, enumerate(data))
    for subject_number, i in iterable_of_graphs:
        print(f"{subject_number} on {len(data)} complete.")
        array_of_graphs.append(i)
end = time.time()
print("Completed in ", end - begin)

DataUtils.data_saver(Config.working_directory + "array_graph_networkx.pickle", array_of_graphs)"""

