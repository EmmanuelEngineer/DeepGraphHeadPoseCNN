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
from tensorflow import keras
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


def generate_model(graphs_for_training, label, eval_graphs, eval_labels):
    generator = PaddedGraphGenerator(graphs=graphs_for_training)
    layer_sizes = [300, 300, 300, 300, 300, 150, 1]
    k = 35
    dgcnn_model = DeepGraphCNN(  # mathematical model that will translate the graph for the next sequence of layers
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "tanh"],
        # for the activation functions refer to https://keras.io/api/layers/activations/
        k=k,  # number of output tensor
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

    predictions = Dense(units=3)(x_out)
    #3 units, 3 outputs(pitch,yaw,roll)

    model_to_fit = Model(inputs=x_inp,
                         outputs=predictions)  # creating the model
    model_to_fit.summary()
    model_to_fit.compile(
        optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=[metrics.mean_squared_error],
        # setting up the optimizer, the loss function (used for training) and the metrics(only to report the advancement of the model, they will not influence the training)
    )

    gen = PaddedGraphGenerator(graphs=graphs_for_training)

    train_gen = gen.flow(  # preparing data for the training
        range(len(graphs_for_training)),
        targets=label,
        batch_size=40,
        symmetric_normalization=False,
    )
    test_generator = PaddedGraphGenerator(graphs=eval_graphs) # preparing data for the evaluation
    test_gen = test_generator.flow(  # dati per il test
        range(len(eval_graphs)),
        targets=eval_labels,
        batch_size=1,
        symmetric_normalization=False,
    )

    epochs = 100  # ripetizioni dell'addestramento

    history_internal = model_to_fit.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=test_gen, shuffle=True,
    )  # fitting the model

    return model_to_fit, history_internal


if __name__ == "__main__":
    for active_weight in Config.all_weight_types:  # for sequential training / i.e. training the models for each
        # metrics in one go
        Config.weight_type = active_weight  # set the current metric
        print("Edge Type:", Config.weight_type)
        labels_list_by_subject = DataUtils.data_loader(Config.working_directory + "labels_by_subject.pickle")

        if Config.weight_type == "euclidean" or Config.weight_type == "ricci":
            all_networkx_list = DataUtils.data_loader(Config.working_directory + "ricci_graphs_by_subject.pickle")
            cleaned_list = []
            for x in all_networkx_list:
                cleaned_list.append(DataUtils.graph_cleaner(x, edge_type=Config.weight_type))  # because the ricci
                # graph function will only add more data to the graph and not erasing it, a cleaning process is needed
            all_networkx_list = cleaned_list

        else:
            all_networkx_list = DataUtils.data_loader(
                Config.working_directory + Config.weight_type + "_graphs_by_subject"
                                                                ".pickle")

        label_list = []

        for x in [x for idx, x in enumerate(labels_list_by_subject)]:
            for y in x:
                label_list.append(y)

        scaler = MinMaxScaler()  # Scaling labels
        training_pandas_labels = pd.DataFrame.from_records(label_list)
        scaler.fit(training_pandas_labels)
        filtered_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
        # removing the subjects that have been recorded multiple type.
        # index based on what is the ImageAnalizer sub-directories output is
        filtered_labels_list_by_subject = [x for idx, x in enumerate(labels_list_by_subject) if idx not in [18, 12]]
        if Config.RegressionSetting.leave_one_subject_out:
            for excluded_subject_id, excluded_subject_networkx_graphs in enumerate(filtered_networkx_list):
                name_file_identifier = Config.weight_type + "_no_" + str(excluded_subject_id) + "_"

                print("prepping for training:", name_file_identifier, " on ", len(filtered_networkx_list))
                training_set = []
                label_list = []
                for x in [x for idx, x in enumerate(filtered_networkx_list) if idx != excluded_subject_id]: # putting all graphs in one list
                    for y in x:
                        training_set.append(y)

                for x in [x for idx, x in enumerate(filtered_labels_list_by_subject) if idx != excluded_subject_id]: # putting all labels in one list
                    for y in x:
                        label_list.append(y)
                print("testingsetlenght:", len(training_set))
                print("label lenght:", len(label_list))
                print(len(training_set[0]))

                print("translating to pandas")
                pandas_graph_list = DataUtils.networkx_list_to_pandas_list(training_set)
                print("translating to stellargraph")
                stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
                print(stellargraph_graphs[0].info())
                training_pandas_labels = pd.DataFrame.from_records(label_list)

                d = scaler.transform(training_pandas_labels)
                training_pandas_labels = pd.DataFrame(d, columns=training_pandas_labels.columns)

                train_graphs, evaluation_graphs, train_labels, evaluation_labels = model_selection.train_test_split(
                    stellargraph_graphs, training_pandas_labels, train_size=0.7, test_size=None, random_state=20
                )

                print("train size", len(train_labels))
                print("evaluation size", len(evaluation_labels))

                model, history = generate_model(train_graphs, train_labels, evaluation_graphs,
                                                evaluation_labels)

                #saving the data in the working directory in a version sub-directory to keep the data after each time you modify something.
                # WARNING create the directory structure BEFORE starting the code
                model.save(
                    Config.working_directory + "v" + str(
                        Config.version) + "/models/" + name_file_identifier + "model.h5")
                DataUtils.data_saver(Config.working_directory + "v" + str(
                    Config.version) + "/histories/" + name_file_identifier + "model_history.pickle",
                                     history)
                DataUtils.data_saver(Config.working_directory + "v" + str(
                    Config.version) + "/scalers/" + name_file_identifier + "result_scaler.pickle",
                                     scaler)
                keras.backend.clear_session()  # to clean the ram after each training

        else: # same thing as before except for the testing set separation
            name_file_identifier = Config.weight_type + "_no_subject_indipendence_"
            print(Config.working_directory + "v" + str(
                Config.version) + "/histories/" + name_file_identifier + "model_history.pickle")
            print("prepping for training:", name_file_identifier)
            dataset = []
            label_list = []
            for x in filtered_networkx_list:
                for y in x:
                    dataset.append(y)

            for x in filtered_labels_list_by_subject:
                for y in x:
                    label_list.append(y)

            print(len(dataset[0]))
            print("translating to pandas")
            pandas_graph_list = DataUtils.networkx_list_to_pandas_list(dataset)
            print("translating to stellargraph")
            stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
            training_pandas_labels = pd.DataFrame.from_records(label_list)

            scaler = MinMaxScaler()  # Scaling labels
            scaler.fit(training_pandas_labels)
            d = scaler.transform(training_pandas_labels)
            training_pandas_labels = pd.DataFrame(d, columns=training_pandas_labels.columns)

            train_graphs, evaluation_graphs, train_labels, evaluation_labels = model_selection.train_test_split(
                stellargraph_graphs, training_pandas_labels, train_size=0.7, test_size=None, random_state=20
            )
            evaluation_graphs, test_graphs, evaluation_labels, test_labels = model_selection.train_test_split(
                evaluation_graphs, evaluation_labels, test_size=0.25, random_state=20
            )
            print("train size", len(train_labels))
            print("evaluation size", len(evaluation_labels))
            print("test size", len(test_labels))

            model, history = generate_model(train_graphs, train_labels, evaluation_graphs, evaluation_labels)

            model.save(
                Config.working_directory + "v" + str(Config.version) + "/models/" + name_file_identifier + "model.h5")
            DataUtils.data_saver(Config.working_directory + "v" + str(
                Config.version) + "/histories/" + name_file_identifier + "model_history.pickle",
                                 history)
            DataUtils.data_saver(Config.working_directory + "v" + str(
                Config.version) + "/scalers/" + name_file_identifier + "result_scaler.pickle",
                                 scaler)
            DataUtils.data_saver(Config.working_directory + "v" + str(
                Config.version) + "/test_graphs/" + name_file_identifier + "testgraphs.pickle",
                                 test_graphs) # saving the test graphs because they are chosen randomly
            DataUtils.data_saver(
                Config.working_directory + "v" + str(
                    Config.version) + "/labels/" + name_file_identifier + "test_labels.pickle",
                pd.DataFrame(scaler.inverse_transform(test_labels), columns=training_pandas_labels.columns))
            keras.backend.clear_session()
