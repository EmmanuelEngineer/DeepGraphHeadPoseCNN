import cv2
import keras.layers
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
from sklearn.preprocessing import MinMaxScaler

from matplotlib import figure
import Config
import networkx as nx

import DataUtils
import Config
import GraphGenerator
import ImageAnalizer


def convert_pandas_graph_list_to_stellargraph(nx_list):
    sg_graphs = []
    for graph in nx_list:  # Conversion to stellargraph Graphs
        sg_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
    return sg_graphs


def generate_model(graphs, oracle):
    generator = PaddedGraphGenerator(graphs=graphs)
    layer_sizes = [75, 75, 1]
    k = 35
    dgcnn_model = DeepGraphCNN(  # Rete neurale che fa da traduttore per quella successiva
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh"],
        # attivazioni applicate all'output del layer: Fare riferimento a https://keras.io/api/layers/activations/
        k=k,  # numero di tensori in output
        bias=False,
        generator=generator,
    )

    x_inp, x_out = dgcnn_model.in_out_tensors()
    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)  # Filtro convoluzionale
    x_out = MaxPool1D(pool_size=2)(x_out)  # https://keras.io/api/layers/pooling_layers/max_pooling1d/

    x_out = Conv1D(filters=32, kernel_size=10, strides=1)(x_out)

    x_out = Flatten()(x_out)

    x_out = Dense(units=512, activation="tanh")(x_out)
    x_out = Dense(units=256, activation="tanh")(x_out)

    x_out = Dropout(rate=0.5)(x_out)
    # Per evitare overfitting setta casualmente degli input a 0 per poi amplificarne il resto per non
    # alterare la somma degli input

    predictions = Dense(units=3)(x_out)

    model = Model(inputs=x_inp,
                  outputs=predictions)  # Setta il modello Keras che effettuerà i calcoli e le predizioni
    model.summary()
    model.compile(
        optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=[metrics.mean_squared_error],
        # Creazione del modello effettivo
    )

    train_graphs, test_graphs = model_selection.train_test_split(
        oracle, train_size=0.7, test_size=None, random_state=20
    )

    gen = PaddedGraphGenerator(graphs=graphs)

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
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    return model, history


if __name__ == "__main__":
    networkx_list = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
    oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)

    if False:#Config.RegressionSetting.apply_RicciCurvature:
        networkx_list = DataUtils.apply_RicciCurvature_on_list(networkx_list)
        DataUtils.data_saver(Config.working_directory + "RiccisCurvatureGraphs.pickle", networkx_list)

    networkx_list = DataUtils.data_loader(Config.working_directory + "RiccisCurvatureGraphs.pickle")
    for graph in networkx_list:
        for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
            for att in ["weight", "original_RC"]:
                d.pop(att, None)

        for n1, d in graph.nodes(data=True):
            for att in ["x", "y", "z"]:
                d.pop(att, None)
    pandas_graph_list = DataUtils.networkx_list_to_pandas_list(networkx_list)
    stellargraph_graphs = convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
    print(stellargraph_graphs[0].info())
    DataUtils.data_saver(Config.working_directory + "stellargraph_graph.pickle", stellargraph_graphs)
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)

    scaler = MinMaxScaler()  # Scaling oracle
    scaler.fit(pandas_oracle)
    d = scaler.transform(pandas_oracle)
    pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

    model, history = generate_model(stellargraph_graphs, pandas_oracle)
    model.save(Config.working_directory + "model.h5")
    DataUtils.data_saver(Config.working_directory+"model_history.pickle", history)

    sg.utils.plot_history(history, return_figure=True).show()

