import glob
import re

import numpy

import Config
import DataUtils
import ImageAnalizer
import stellargraph as sg
import tensorflow
from tensorflow import keras
from keras import Sequential
import matplotlib as plt

import pandas as pd


def get_object_data_from_path(path):
    # getting string identifier, index of the excluded subject and metric/weight type
    print("path", path)
    file_data = re.findall("\/([a-z]+)_no_(\d*)|(subject_indipendence)_model\.h5/gm", path)
    print("file Data", file_data)
    weight_type = file_data[0][0]
    if file_data[0][1] != "":
        number_id = int(file_data[0][1])
    else:
        number_id = None
    if number_id is None:
        string_identifier = file_data[0][0]+"_no_subject_indipendence_"
    else:
        string_identifier = file_data[0][0] + "_no_" + str(number_id)+"_"

    return string_identifier, weight_type, number_id


if __name__ == "__main__":
    from stellargraph.layer import GraphConvolution, SortPooling
    from stellargraph.mapper import PaddedGraphGenerator
    from keras.saving.save import load_model

    model_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/models/*")
    histories_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/histories/*")
    scalers_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/scalers/*")

    euclidean_graphs = DataUtils.data_loader(Config.working_directory + "/euclidean_graphs_by_subject.pickle")
    ricci_graphs = DataUtils.data_loader(Config.working_directory + "/ricci_graphs_by_subject.pickle")
    cityblock_graphs = DataUtils.data_loader(Config.working_directory + "/cityblock_graphs_by_subject.pickle")
    cosine_graphs = DataUtils.data_loader(Config.working_directory+"/cosine_graphs_by_subject.pickle")
    label_list_by_subject = DataUtils.data_loader(Config.working_directory + "/labels_by_subject.pickle")

    cosine_graphs = [x for idx, x in enumerate(cosine_graphs) if idx not in [18, 12]]  # same exclusion as the RegressorGenerator
    cityblock_graphs = [x for idx, x in enumerate(cityblock_graphs) if idx not in [18, 12]]
    euclidean_graphs = [x for idx, x in enumerate(euclidean_graphs) if idx not in [18, 12]]
    ricci_graphs = [x for idx, x in enumerate(ricci_graphs) if idx not in [18, 12]]

    label_list_by_subject = [x for idx, x in enumerate(label_list_by_subject) if idx not in [18, 12]]
    custom_objects = {"GraphConvolution": GraphConvolution, "SortPooling": SortPooling}
    # loading custom stellargraph objects used for the creation of the regrassion

    for model_path in model_paths:
        identifier, edge_type, number_of_model = get_object_data_from_path(model_path)
        model = load_model(model_path, custom_objects=custom_objects)

        history = None
        for history_path in histories_paths:  # get history data for the model
            if re.findall(identifier, history_path):
                history = DataUtils.data_loader(history_path)
                break

        scaler = None
        for scaler_path in scalers_paths: # get MinMaxScaler for the model
            if re.findall(identifier, scaler_path):
                scaler = DataUtils.data_loader(scaler_path)
                break

        plot = sg.utils.plot_history(history, return_figure=True)
        plot.savefig(Config.working_directory + "v" + str(Config.version) + "/history_plots/" + identifier + ".jpg") # saving history plot of the model
        plot.clear()
        if number_of_model is not None:
            pandas_labels = pd.DataFrame.from_records(label_list_by_subject[number_of_model])
            if edge_type == "ricci": # selecting the graphs needed for the model
                testing_graphs = DataUtils.graph_cleaner(ricci_graphs[number_of_model], edge_type="ricci")

            elif edge_type == "cosine":
                testing_graphs = cosine_graphs[number_of_model]
            elif edge_type == "euclidean":
                testing_graphs = DataUtils.graph_cleaner(ricci_graphs[number_of_model], edge_type="euclidean")

            else: testing_graphs = cityblock_graphs[number_of_model]

            pandas_graph_list = DataUtils.networkx_list_to_pandas_list(testing_graphs)
            stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)

        else:
            stellargraph_graphs = DataUtils.data_loader(
                Config.working_directory + "v" + str(Config.version) + "/test_graphs"
                                                                       "/" + identifier + "testgraphs.pickle")

            pandas_labels = DataUtils.data_loader(
                Config.working_directory + "v" + str(Config.version) + "/labels"
                                                                       "/" + identifier + "test_labels.pickle")

        generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
        # converting data to be predicted on using a generator

        obj = generator.flow(stellargraph_graphs)
        try:
            non_scaled_prediction = model.predict(obj)
        except Exception as msg:
            print(msg)
            continue

        print("non_scaled_prediction", non_scaled_prediction)
        scaled_prediction = []
        for x in non_scaled_prediction:
            scaled_prediction.append(scaler.inverse_transform(x.reshape(1, -1)))
        scaled_prediction = numpy.concatenate(scaled_prediction)
        pandas_predictions = pd.DataFrame(scaled_prediction,
                                          columns=("predicted_pitch", "predicted_yaw", "predicted_roll"))
        pandas_predictions = pandas_predictions.join(pandas_labels)
        pandas_predictions.to_csv(
            Config.working_directory + "v" + str(Config.version) + "/predictions/" + identifier + ".csv") # saving the predicted values
        keras.backend.clear_session()
