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


def generate_model(graphs_for_training, oracle, testing_graphs, testing_oracle):
    print("train_graphs", len(graphs_for_training))
    print("train_oracle", oracle)
    print("validation_graphs", testing_graphs)
    print("validation_oracle",testing_oracle)
    generator = PaddedGraphGenerator(graphs=graphs_for_training)
    layer_sizes = [300, 300, 300, 300, 300, 150, 1]
    k = 35
    dgcnn_model = DeepGraphCNN(  # Rete neurale che fa da traduttore per quella successiva
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "tanh"],        # attivazioni applicate all'output del layer: Fare riferimento a https://keras.io/api/layers/activations/
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

    gen = PaddedGraphGenerator(graphs=graphs_for_training)

    train_gen = gen.flow(  # dati per l'addestramento
        range(len(graphs_for_training)),
        targets=oracle,
        batch_size=40,
        symmetric_normalization=False,
    )
    test_generator = PaddedGraphGenerator(graphs=testing_graphs)
    test_gen = test_generator.flow(  # dati per il test
        range(len(testing_graphs)),
        targets=testing_oracle,
        batch_size=1,
        symmetric_normalization=False,
    )

    epochs = 400  # ripetizioni dell'addestramento

    history_internal = model_to_fit.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
    )
    test_metrics = model_to_fit.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model_to_fit.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))

    return model_to_fit, history_internal


if __name__ == "__main__":
    oracle_list_by_subject = DataUtils.data_loader(Config.working_directory + "oracle_by_subject.pickle")
    if False:  # Config.RegressionSetting.apply_RicciCurvature:
        networkx_list_by_subject = DataUtils.apply_RicciCurvature_on_list(training_set)
        DataUtils.data_saver(Config.working_directory + "oracle_by_subject.pickle", training_set)

    all_networkx_list = DataUtils.data_loader(Config.working_directory + "ricci_by_subject_def.pickle")
    cleaned_list = []
    for x in all_networkx_list:
        cleaned_list.append(DataUtils.graph_cleaner(x))

    oracle_list = []
    all_networkx_list = cleaned_list
    for x in [x for idx, x in enumerate(oracle_list_by_subject)]:
        for y in x:
            oracle_list.append(y)

    scaler = MinMaxScaler()  # Scaling oracle
    pandas_oracle = pd.DataFrame.from_records(oracle_list)
    scaler.fit(pandas_oracle)
    all_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
    oracle_list_by_subject = [x for idx, x in enumerate(oracle_list_by_subject) if idx not in [18, 12]]
    if Config.RegressionSetting.subject_indipendence:
        for excluded_subject_id, excluded_subject_networkx_graphs in enumerate(all_networkx_list):

            name_file_identifier = "no_"+str(excluded_subject_id)+"_" if not Config.RegressionSetting.apply_RicciCurvature else "ricci_no_"+str(excluded_subject_id)+"_"

            print("prepping for training:", name_file_identifier, " on ", len(all_networkx_list))
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
            stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
            testing_stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(testing_pandas_graph_list)
            print(stellargraph_graphs[0].info())
            pandas_oracle = pd.DataFrame.from_records(oracle_list)
            testing_pandas_oracle = pd.DataFrame.from_records(oracle_list_by_subject[excluded_subject_id])

            d = scaler.transform(pandas_oracle)
            pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

            d = scaler.transform(testing_pandas_oracle)
            testing_pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

            model, history = generate_model(stellargraph_graphs, pandas_oracle, testing_stellargraph_graphs, testing_pandas_oracle)

            model.save(Config.working_directory +"v"+str(Config.version)+ "/models/" + name_file_identifier + "model.h5")
            DataUtils.data_saver(Config.working_directory +"v"+str(Config.version)+ "/histories/" + name_file_identifier + "model_history.pickle",
                                 history)
            DataUtils.data_saver(Config.working_directory +"v"+str(Config.version)+ "/scalers/" + name_file_identifier + "result_scaler.pickle",
                                 scaler)

    else:
        name_file_identifier = "no_subject_indipendence_" if not Config.RegressionSetting.apply_RicciCurvature else "ricci_no_subject_indipendence_"
        print(Config.working_directory +"v"+str(Config.version)+ "/histories/" + name_file_identifier + "model_history.pickle")
        print("prepping for training:", name_file_identifier)
        dataset = []
        oracle_list = []
        for x in all_networkx_list:
            for y in x:
                dataset.append(y)

        for x in oracle_list_by_subject:
            for y in x:
                oracle_list.append(y)

        print("testing set lenght:", len(dataset))
        print("oracle lenght:", len(oracle_list))

        print("translating to pandas")
        pandas_graph_list = DataUtils.networkx_list_to_pandas_list(dataset)
        print("translating to stellargraph")
        stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
        pandas_oracle = pd.DataFrame.from_records(oracle_list)

        scaler = MinMaxScaler()  # Scaling oracle
        scaler.fit(pandas_oracle)
        d = scaler.transform(pandas_oracle)
        pandas_oracle = pd.DataFrame(d, columns=pandas_oracle.columns)

        train_graphs, test_graphs, train_oracle, test_oracle = model_selection.train_test_split(
            stellargraph_graphs, pandas_oracle, train_size=0.7, test_size=None, random_state=20
        )

        model, history = generate_model(train_graphs, train_oracle, test_graphs, test_oracle)

        model.save(Config.working_directory +"v"+str(Config.version)+ "/models/" + name_file_identifier + "model.h5")
        DataUtils.data_saver(Config.working_directory +"v"+str(Config.version)+ "/histories/" + name_file_identifier + "model_history.pickle",
                             history)
        DataUtils.data_saver(Config.working_directory +"v"+str(Config.version)+ "/scalers/" + name_file_identifier + "result_scaler.pickle",
                             scaler)
        DataUtils.data_saver(Config.working_directory+"v"+str(Config.version) + "/testgraphs/" + name_file_identifier + "testgraphs.pickle",
                             test_graphs)
        DataUtils.data_saver(
            Config.working_directory+"v"+str(Config.version) + "/oracles/" + name_file_identifier + "test_oracle.pickle",
            pd.DataFrame(scaler.inverse_transform(test_oracle), columns=pandas_oracle.columns))






