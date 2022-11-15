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
        ricci = False
        data_model = re.findall("(ricci_)no_(\d+)_", model_path)
        print(data_model)
        if "ricci_" in model_path:
            ricci= True
        number_of_model = None
        if ricci:
            number_of_model = int(data_model[0][1])
        else: number_of_model = int(data_model[0][0])
        history = None
        identifier = ("ricci_" if ricci else "") + "no_"+str(number_of_model)

        for history_path in histories_paths:
            if re.findall(identifier, history_path):
                history = DataUtils.data_loader(history_path)
                break

        scaler = None
        for scaler_path in scalers_paths:
            if re.findall(identifier, scaler_path):
                scaler = DataUtils.data_loader(scaler_path)
                break



        #DataUtils.plot_history_scaled(history, scaler, return_figure=True).show()
        sg.utils.plot_history(history, return_figure=True).show()
        testing_graphs = DataUtils.graph_cleaner(all_networkx_list[number_of_model], ricci_graphs=ricci)
        pandas_graph_list = DataUtils.networkx_list_to_pandas_list(testing_graphs)
        stellargraph_graphs = DataUtils.convert_pandas_graph_list_to_stellargraph(pandas_graph_list)
        generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
        obj = generator.flow(stellargraph_graphs)
        non_scaled_prediction = model.predict(obj)
        scaled_prediction = scaler.inverse_transform(non_scaled_prediction)
        pandas_oracle = pd.DataFrame.from_records(oracle_list_by_subject[number_of_model])
        pandas_predictions = pd.DataFrame(scaled_prediction, columns=("predicted pitch", "predicted yaw", "predicted roll"))
        pandas_predictions = pandas_predictions.join(pandas_oracle)
        pandas_predictions.to_csv(Config.working_directory + "predictions/"+identifier+".csv")


