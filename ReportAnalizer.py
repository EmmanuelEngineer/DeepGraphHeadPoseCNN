import glob

import Config
import DataUtils
import ModelTester
import pandas as pd
import matplotlib as plt


if __name__ == "__main__":
    predictions_paths = glob.glob(Config.working_directory + "/predictions/*")
    prediction_ricci_no_subject_indipendence = None
    prediction_no_subject_indipendence = None
    prediction_subject_indipendence = []
    prediction_ricci_subject_indipendence = []
    for prediction_path in predictions_paths:
        identifier, ricci, number_of_model = ModelTester.get_object_data_from_path(prediction_path)
        if number_of_model:
            if ricci:
                prediction_ricci_no_subject_indipendence = pd.read_csv(prediction_path)
            else:
                prediction_no_subject_indipendence = pd.read_csv(prediction_path)
        else:
            if ricci:
                prediction_ricci_subject_indipendence.append(pd.read_csv(prediction_path))
            else:
                prediction_subject_indipendence.append(pd.read_csv(prediction_path))




