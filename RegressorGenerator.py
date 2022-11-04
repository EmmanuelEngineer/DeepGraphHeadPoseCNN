import cv2
import pandas as pd
import pandas as pd
import numpy as np

import stellargraph as sg
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
import torch
from sklearn import model_selection
import tensorrt
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.initializers import Zeros
from keras import metrics
from matplotlib import figure
import Config
import networkx as nx

import DataUtils
import Config
import GraphGenerator
import ImageAnalizer
"""
networkx_list = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")

pandas_oracle = pd.DataFrame.from_dict(oracle_list)
pandas_graph_list = DataUtils.networkx_list_to_pandas_list(networkx_list)
print(pandas_oracle)

stellargraph_graphs = []
for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
    stellargraph_graphs.append(StellarGraph(
        {"landmark": graph["nodes"]}, {"line": graph["edges"]}))


DataUtils.data_saver(Config.working_directory + "stellargraph_graph.pickle", stellargraph_graphs)
"""

print(torch.version.cuda)
oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")
pandas_oracle = pd.DataFrame.from_dict(oracle_list)
stellargraph_graphs = DataUtils.data_loader(Config.working_directory + "stellargraph_graph.pickle")
print(stellargraph_graphs[0].info())
generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
layer_sizes = [500, 500, 500, 1]
k = 80
dgcnn_model = DeepGraphCNN(  # Rete neurale che fa da traduttore per quella successiva
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    # attivazioni applicate all'output del layer: Fare riferimento a https://keras.io/api/layers/activations/
    k=k,  # numero di tensori in output
    bias=False,
    generator=generator,
)

# Rete Neurale che effettua il lavoro
x_inp, x_out = dgcnn_model.in_out_tensors()
x_out = Conv1D(filters=18, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)  # Filtro convoluzionale
x_out = MaxPool1D(pool_size=2)(x_out)  # https://keras.io/api/layers/pooling_layers/max_pooling1d/

x_out = Conv1D(filters=32, kernel_size=10, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=204, activation="tanh")(x_out)
x_out = Dropout(rate=0.5)(x_out)
# Per evitare overfitting setta casualmente degli input a 0 per poi amplificarne il resto per non
# alterare la somma degli input


predictions = Dense(units=1)(x_out)

model = Model(inputs=x_inp, outputs=predictions)  # Setta il modello Keras che effettuerà i calcoli e le predizioni

model.compile(
    optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=[metrics.mean_absolute_percentage_error],
    # Creazione del modello effettivo
)

train_graphs, test_graphs = model_selection.train_test_split(
    pandas_oracle["pitch"], train_size=0.7, test_size=None, random_state=20
)

gen = PaddedGraphGenerator(graphs=stellargraph_graphs)

print(train_graphs)
train_gen = gen.flow(  # dati per l'addestramento
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=40,
    symmetric_normalization=False,
)

test_gen = gen.flow(  # dati per il test
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

epochs = 100  # ripetizioni dell'addestramento

history = model.fit(
    train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
)
"""
print(sg.utils.plot_history(history))

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print(model.predict(test_gen))
"""
model.save(Config.working_directory + "model.h5")

images_paths = ImageAnalizer.import_images_paths(Config.Test.image_dataset)
point_array = ImageAnalizer.landmark_extraction(images_paths)

print(images_paths)

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

    pandas_graph_list = DataUtils.networkx_list_to_pandas_list([graph])
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)
    pandas_graph_list[0]["nodes"].to_csv("nodes.csv")
    pandas_graph_list[0]["edges"].to_csv("edges.csv")

    stellargraph_graphs = []
    for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
        stellargraph_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
    test_generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
    obj = test_generator.flow(stellargraph_graphs)
    predict = model.predict(obj)
    print("predicted", predict, "expected", oracle_list[0])
