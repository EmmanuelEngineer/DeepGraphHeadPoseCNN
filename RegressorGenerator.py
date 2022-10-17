import pandas as pd
import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from sklearn import model_selection

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from keras import metrics

import Config
import networkx as nx

import DataUtils

newtorkx_list = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")


def networkx_list_to_pandas_list(input_list):  # Step taken for efficiency of as of the stellargraph docs
    output_list = []
    for x in input_list:
        edges = nx.to_pandas_edgelist(x)
        nodes = pd.DataFrame.from_dict(dict(x.nodes(data=True)), orient='index')
        output_list.append({"nodes": nodes, "edges": edges})
    return output_list

"""
pandas_oracle = pd.DataFrame.from_dict(oracle_list)
pandas_graph_list = networkx_list_to_pandas_list(newtorkx_list)
print(pandas_oracle)

stellargraph_graphs = []
for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
    stellargraph_graphs.append(StellarGraph(
        {"landmark": graph["nodes"]}, {"line": graph["edges"]}))

print(stellargraph_graphs[0].info())

generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
layer_sizes = [32, 32, 32, 3]
k = 35
dgcnn_model = DeepGraphCNN(  #  Rete neurale che fa da traduttore per quella successiva
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    # attivazioni applicate all'output del layer: Fare riferimento a https://keras.io/api/layers/activations/
    k=k,
    bias=False,
    generator=generator,
)

# Rete Neurale per la traduzione
x_inp, x_out = dgcnn_model.in_out_tensors()
x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)  # Filtro convoluzionale
x_out = MaxPool1D(pool_size=2)(x_out)  # https://keras.io/api/layers/pooling_layers/max_pooling1d/

x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=128, activation="relu")(x_out)
x_out = Dropout(rate=0.5)(x_out)
# Per evitare overfitting setta casualmente degli input a 0 per poi amplificarne il resto per non
# alterare la somma degli input

predictions = Dense(units=3)(x_out)

model = Model(inputs=x_inp, outputs=predictions)  # Setta il modello Keras che effettuerà i calcoli e le predizioni

model.compile(
    optimizer=Adam(lr=0.0001), loss="mean_squared_error", metrics=[metrics.cosine_proximity],  # Creazione del modello effettivo
)

train_graphs, test_graphs = model_selection.train_test_split(
    pandas_oracle, train_size=0.9, test_size=None, random_state=20
)

gen = PaddedGraphGenerator(graphs=stellargraph_graphs)

print(train_graphs)
train_gen = gen.flow(  # dati per l'addestramento
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=20,
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

sg.utils.plot_history(history)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

model.save("model.h5")
"""