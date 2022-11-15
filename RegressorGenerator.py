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
    for idx, graph_nx in enumerate(nx_list):  # Conversion to stellargraph Graphs
        sg_graphs.append(StellarGraph(
            {"landmark": graph_nx["nodes"]}, {"line": graph_nx["edges"]}))
    return sg_graphs


def generate_model(graphs, oracle, testing_graphs, testing_oracle):
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

    model_to_fit = Model(inputs=x_inp,
                         outputs=predictions)  # Setta il modello Keras che effettuerà i calcoli e le predizioni
    model_to_fit.summary()
    model_to_fit.compile(
        optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=[metrics.mean_squared_error],
        # Creazione del modello effettivo
    )




    gen = PaddedGraphGenerator(graphs=graphs)

    train_gen = gen.flow(  # dati per l'addestramento
        range(len(training_set)),
        targets=oracle,
        batch_size=40,
        symmetric_normalization=False,
    )
    gen = PaddedGraphGenerator(graphs=testing_graphs)
    test_gen = gen.flow(  # dati per il test
        range(len(testing_graphs)),
        targets=testing_oracle,
        batch_size=1,
        symmetric_normalization=False,
    )

    epochs = 100  # ripetizioni dell'addestramento

    history_internal = model_to_fit.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
    )
    test_metrics = model_to_fit.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model_to_fit.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    return model_to_fit, history_internal


if __name__ == "__main__":
    networkx_list_by_subject = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
    oracle_list_by_subject = DataUtils.data_loader(Config.working_directory + "oracle_by_subject.pickle")
    if False:  # Config.RegressionSetting.apply_RicciCurvature:
        networkx_list_by_subject = DataUtils.apply_RicciCurvature_on_list(training_set)
        DataUtils.data_saver(Config.working_directory + "oracle_by_subject.pickle", training_set)

    all_networkx_list = DataUtils.data_loader(Config.working_directory + "ricci_by_subject_corrected.pickle")

    for training_set in all_networkx_list:
        if Config.RegressionSetting.apply_RicciCurvature:
            for graph in training_set:
                for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
                    for att in ["weight", "original_RC"]:
                        d.pop(att, None)

                for n1, d in graph.nodes(data=True):
                    for att in ["x", "y", "z"]:
                        d.pop(att, None)
        else:
            for graph in training_set:
                for n1, n2, d in graph.edges(data=True):  # leaving only the ricciscurvature result as weight
                    for att in ["ricciCurvature", "original_RC"]:
                        d.pop(att, None)

                for n1, d in graph.nodes(data=True):
                    for att in ["ricciCurvature"]:
                        d.pop(att, None)

    all_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
    oracle_list_by_subject = [x for idx, x in enumerate(oracle_list_by_subject) if idx not in [18, 12]]

    for excluded_subject_id, excluded_subject_networkx_graphs in enumerate(all_networkx_list):
        print("prepping for training:", excluded_subject_id, " on ", len(all_networkx_list))
        training_set = []
        oracle_list = []
        for x in [x for idx, x in enumerate(all_networkx_list) if idx != excluded_subject_id]:
            for y in x:
                training_set.append(y)

        for x in [x for idx, x in enumerate(oracle_list_by_subject) if idx != excluded_subject_id]:
            for y in x:
                oracle_list.append(y)
        print("testingsetlenght:", len(training_set))
        print("oracle lenght:", len(oracle_list))

        print("translating to pandas")
        pandas_graph_list = DataUtils.networkx_list_to_pandas_list(training_set)
        testing_pandas_graph_list = DataUtils.networkx_list_to_pandas_list(excluded_subject_networkx_graphs)
        print("translating to stellargraph")
        stellargraph_graphs = convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
        testing_stellargraph_graphs = convert_pandas_graph_list_to_stellargraph(testing_pandas_graph_list)
        print(stellargraph_graphs[0].info())
        pandas_oracle = pd.DataFrame.from_records(oracle_list)
        testing_pandas_oracle = pd.DataFrame.from_records(oracle_list_by_subject[excluded_subject_id])

        scaler = MinMaxScaler()  # Scaling oracle
        scaler.fit(pandas_oracle)
        d = scaler.transform(pandas_oracle)
        pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

        d = scaler.transform(testing_pandas_oracle)
        testing_pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

        model, history = generate_model(stellargraph_graphs, pandas_oracle, testing_stellargraph_graphs, testing_pandas_oracle )
        name_file_identifier = "no_"+str(excluded_subject_id)+"_" if not Config.RegressionSetting.apply_RicciCurvature else "ricci_no_"+str(excluded_subject_id)+"_"

        model.save(Config.working_directory + "/models/" + name_file_identifier + "model.h5")
        DataUtils.data_saver(Config.working_directory + "/histories/" + name_file_identifier + "model_history.pickle",
                             history)
        DataUtils.data_saver(Config.working_directory + "/scalers/" + name_file_identifier + "result_scaler.pickle",
                             scaler)
