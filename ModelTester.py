import glob
import re

import Config
import DataUtils
import ImageAnalizer
import stellargraph as sg
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.saving.save import load_model

if __name__ == "__main__":
    model_paths = glob.glob(Config.working_directory + "/models/")
    histories_paths = glob.glob(Config.working_directory + "/histories/")
    scalers_paths = glob.glob(Config.working_directory + "/scalers/")

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
    oracle_list_by_subject = DataUtils.data_loader(Config.working_directory + "oracle_by_subject.pickle")

    all_networkx_list = [x for idx, x in enumerate(all_networkx_list) if idx not in [18, 12]]
    oracle_list_by_subject = [x for idx, x in enumerate(oracle_list_by_subject) if idx not in [18, 12]]

    for model_path in enumerate(model_paths):
        ricci = False
        data_model = re.findall("(ricci_)no_(\d)_", model_path)
        if "ricci_" in model_path:
            ricci= True
        number_of_model = None
        if ricci:
            number_of_model = data_model[1]
        else: number_of_model = data_model[0]
        history = None
        identifier = ("ricci_" if ricci else "") + "no_"+str(number_of_model)

        for history_path in histories_paths:
            if re.findall(identifier, history_path):
                history = DataUtils.data_loader(history_path)
                break

        for scaler_path in scalers_paths:
            if re.findall(identifier, scaler_path):
                scaler = DataUtils.data_loader(scaler_path)
                break

        DataUtils.plot_history_scaled(history, scaler, return_figure=True).show()
