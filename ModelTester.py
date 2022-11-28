import glob
import re

import numpy
from stellargraph.layer import GraphConvolution, SortPooling
from stellargraph.mapper import PaddedGraphGenerator

import Config
import DataUtils
import ImageAnalizer
import stellargraph as sg
import tensorflow
from tensorflow import keras
from keras import Sequential
import matplotlib as plt
from keras.saving.save import load_model
import pandas as pd


def get_object_data_from_path(path):
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
    model_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/models/*")
    histories_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/histories/*")
    scalers_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/scalers/*")

    all_networkx_list = DataUtils.data_loader(Config.working_directory + "/ricci_graphs_by_subject.pickle")
    label_list_by_subject = DataUtils.data_loader(Config.working_directory + "/labels_by_subject.pickle")

    all_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
    label_list_by_subject = [x for idx, x in enumerate(label_list_by_subject) if idx not in [18, 12]]
    custom_objects = {"GraphConvolution": GraphConvolution, "SortPooling": SortPooling}

    for model_path in model_paths:
        identifier, edge_type, number_of_model = get_object_data_from_path(model_path)
        model = load_model(model_path, custom_objects=custom_objects)

        history = None
        for history_path in histories_paths:
            if re.findall(identifier, history_path):
                history = DataUtils.data_loader(history_path)
                break

        scaler = None
        for scaler_path in scalers_paths:
            if re.findall(identifier, scaler_path):
                scaler = DataUtils.data_loader(scaler_path)
                break

        plot = sg.utils.plot_history(history, return_figure=True)
        plot.savefig(Config.working_directory + "v" + str(Config.version) + "/history_plots/" + identifier + ".jpg")
        plot.clear()
        if number_of_model is not None:
            pandas_labels = pd.DataFrame.from_records(label_list_by_subject[number_of_model])
            if edge_type == "ricci" or edge_type == "euclidean":
                testing_graphs = DataUtils.graph_cleaner(all_networkx_list[number_of_model], edge_type=edge_type)
            else:
                testing_graphs = all_networkx_list[number_of_model]
            pandas_graph_list = DataUtils.networkx_list_to_pandas_list(testing_graphs)
            if pandas_graph_list[0]["nodes"].empty:
                print("skipping", identifier)
                continue
            stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)

        else:
            stellargraph_graphs = DataUtils.data_loader(
                Config.working_directory + "v" + str(Config.version) + "/test_graphs"
                                                                       "/" + identifier + "testgraphs.pickle")

            pandas_labels = DataUtils.data_loader(
                Config.working_directory + "v" + str(Config.version) + "/labels"
                                                                       "/" + identifier + "test_labels.pickle")

        generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
        obj = generator.flow(stellargraph_graphs)
        non_scaled_prediction = model.predict(obj)
        print("non_scaled_prediction", non_scaled_prediction)
        scaled_prediction = []
        for x in non_scaled_prediction:
            scaled_prediction.append(scaler.inverse_transform(x.reshape(1, -1)))
        scaled_prediction = numpy.concatenate(scaled_prediction)
        pandas_predictions = pd.DataFrame(scaled_prediction,
                                          columns=("predicted_pitch", "predicted_yaw", "predicted_roll"))
        pandas_predictions = pandas_predictions.join(pandas_labels)
        pandas_predictions.to_csv(
            Config.working_directory + "v" + str(Config.version) + "/predictions/" + identifier + ".csv")
