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

if __name__ == "__main__":
    networkx_list = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
    oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)
    pandas_graph_list = DataUtils.networkx_list_to_pandas_list(networkx_list)

    if False:
        if Config.RegressionSetting.apply_RicciCurvature:
            networkx_list = DataUtils.apply_RicciCurvature_on_list(networkx_list)
            DataUtils.data_saver(Config.working_directory + "RiccisCurvatureGraphs.pickle", networkx_list)

        ricci_networkx_list = DataUtils.data_loader(Config.working_directory + "RiccisCurvatureGraphs.pickle")

        print(pandas_oracle)
        for graph in ricci_networkx_list:
            for n1, n2, d in graph.edges(data=True):
                for att in ["weight", "original_RC"]:
                    d.pop(att, None)

            for n1, d in graph.nodes(data=True):
                d.pop("ricciCurvature", None)

    ##
    stellargraph_graphs = []
    for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
        stellargraph_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))

    DataUtils.data_saver(Config.working_directory + "stellargraph_graph.pickle", stellargraph_graphs)
    ##

    stellargraph_graphs = DataUtils.data_loader(Config.working_directory + "stellargraph_graph.pickle")
    print(torch.version.cuda)
    # Scaling oracle
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)

    scaler = MinMaxScaler()
    scaler.fit(pandas_oracle)
    d = scaler.transform(pandas_oracle)
    pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)
    DataUtils.data_saver(Config.working_directory + "minmaxscaled_oracle.pickle", pandas_oracle)

    print(stellargraph_graphs[0].info())
    generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
    layer_sizes = [150, 150, 150, 150, 150, 150, 1]
    k = 35
    dgcnn_model = DeepGraphCNN(  # Rete neurale che fa da traduttore per quella successiva
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "tanh"],
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
        pandas_oracle, train_size=0.7, test_size=None, random_state=20
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

    epochs = 200  # ripetizioni dell'addestramento

    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
    )
    model.save(Config.working_directory + "model.h5")

    print(sg.utils.plot_history(history))
    image = np.zeros((1000, 1000, 3), dtype="uint8")
    plt.imshow(image)
    plt.show()

    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    print(model.predict(test_gen))
