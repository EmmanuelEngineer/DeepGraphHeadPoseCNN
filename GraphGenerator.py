import DataUtils
from config import Config
import networkx as nx
import math


data = DataUtils.data_loader(Config.dataset_folder + "data_array.pickle")


def node_distance(a, b):
    d = math.sqrt(math.pow(b["x"] - a["x"], 2) +
                  math.pow(b["y"] - a["y"], 2) +
                  math.pow(b["z"] - a["z"], 2) * 1.0)
    return d


def to_ignore(a_index, b_index, selected_edges):
    counter = 0
    flag = False
    for edge in selected_edges:
        if edge[0] == a_index and edge[1] == b_index:  # The opposite edge exist
            flag = False
            break

        if edge[1] == b_index or edge[0] == b_index:
            counter += 1

    if counter >= Config.Extraction.edges_per_index - 1 or flag:
        return True
    else:
        return False


def sort(edge):
    return edge[2]


def edge_list_generator(node_list):
    selected_edges = []
    for a_index, a_node in enumerate(node_list):
        edges_per_single_node = []
        counter = 0
        if a_index == 104:
            print(number_of_edges, " ", edges_per_single_node , selected_edges)
        for edge in selected_edges:
            if edge[1] == a_index or edge[0] == a_index:  # number of edges from/to a node
                counter += 1
        number_of_edges = Config.Extraction.edges_per_index - counter
        if number_of_edges <= 0:
            continue
        for b_index, b_node in enumerate(node_list):
            if a_index == b_index:
                continue

            if not to_ignore(a_index, b_index, selected_edges):
                edges_per_single_node.append((a_index, b_index,
                                              node_distance(node_list[a_index], node_list[b_index])))

        #  minimum_edges = edges_per_single_node[0:Config.Extraction.edges_per_index]
        edges_per_single_node.sort(key=sort)

        for x in edges_per_single_node[:number_of_edges]:
            selected_edges.append(x)
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
        print(f"{subject_number} on {len(data) - 1} complete.")
        array_of_graphs.append(i)
end = time.time()
print("Completed in ", end - begin)
print(array_of_graphs)

DataUtils.data_saver(Config.dataset_folder + "array_graph_networkx.pickle", array_of_graphs)
"""
