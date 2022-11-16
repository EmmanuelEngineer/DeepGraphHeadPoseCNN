import glob
import re

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
    is_ricci = False
    file_data = re.findall("(ricci_)no_(\d+)_", model_path)
    print(file_data)
    if len(file_data) != 0:
        if "ricci_" in model_path:
            is_ricci = True
        number_id = None
        if is_ricci:
            number_id = int(file_data[0][1])
        else:
            number_id = int(file_data[0][0])
        string_identifier = ("ricci_" if is_ricci else "") + "no_" + str(number_id)
    else:
        file_data = re.findall("(ricci_)no_subject_indipendence_", model_path)
        if "ricci_" in model_path:
            is_ricci = True
        number_id = None
        string_identifier = ("ricci_" if is_ricci else "") + "no_subject_indipendence_"

    return string_identifier, is_ricci, number_id


if __name__ == "__main__":
    model_paths = glob.glob(Config.working_directory + "/models/*")
    histories_paths = glob.glob(Config.working_directory + "/histories/*")
    scalers_paths = glob.glob(Config.working_directory + "/scalers/*")

    all_networkx_list = DataUtils.data_loader(Config.working_directory + "ricci_by_subject_corrected.pickle")
    oracle_list_by_subject = DataUtils.data_loader(Config.working_directory + "oracle_by_subject.pickle")

    all_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
    oracle_list_by_subject = [x for idx, x in enumerate(oracle_list_by_subject) if idx not in [18, 12]]
    custom_objects = {"GraphConvolution": GraphConvolution, "SortPooling": SortPooling}
    for model_path in model_paths:
        print(model_path)
        model = load_model(model_path, custom_objects=custom_objects)

        identifier, ricci, number_of_model = get_object_data_from_path(model_path)

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

        # DataUtils.plot_history_scaled(history, scaler, return_figure=True).show()
        plot = sg.utils.plot_history(history, return_figure=True)
        plot.savefig(Config.working_directory + "history_plots/" + identifier + ".jpg")
        plot.clear()
        testing_graphs = DataUtils.graph_cleaner(all_networkx_list[number_of_model], ricci_graphs=ricci)
        pandas_graph_list = DataUtils.networkx_list_to_pandas_list(testing_graphs)
        stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
        generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
        obj = generator.flow(stellargraph_graphs)
        non_scaled_prediction = model.predict(obj)
        scaled_prediction = scaler.inverse_transform(non_scaled_prediction)
        pandas_oracle = pd.DataFrame.from_records(oracle_list_by_subject[number_of_model])
        pandas_predictions = pd.DataFrame(scaled_prediction,
                                          columns=("predicted pitch", "predicted yaw", "predicted roll"))
        pandas_predictions = pandas_predictions.join(pandas_oracle)
        pandas_predictions.to_csv(Config.working_directory + "predictions/" + identifier + ".csv")
