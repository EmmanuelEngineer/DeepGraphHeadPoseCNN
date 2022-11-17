import glob

import numpy as np

import Config
import DataUtils
import ModelTester
import pandas as pd
import matplotlib as plt


def extract_data_from_report(report):
    max_list = report[["pitch", "yaw", "roll"]].abs().max(axis=1)
    report["distance_from_center"] = max_list.values

    error_pitch_eval = "pitch_error = pitch - predicted_pitch"
    error_yaw_eval = "yaw_error = yaw - predicted_yaw"
    error_roll_eval = "roll_error = roll - predicted_roll"

    report = pd.eval(error_pitch_eval, target=report)
    report = pd.eval(error_yaw_eval, target=report)
    report = pd.eval(error_roll_eval, target=report)

    report["pitch_error"] = report["pitch_error"].abs()
    report["yaw_error"] = report["yaw_error"].abs()
    report["roll_error"] = report["roll_error"].abs()

    max_list = report[["pitch_error", "yaw_error", "roll_error"]].abs().max(axis=1)
    report["max_error"] = max_list.values
    max_list = report[["pitch_error", "yaw_error", "roll_error"]].abs().idxmax(axis=1)
    report["max_error_axis"] = max_list.values

    return report


if __name__ == "__main__":
    predictions_paths = glob.glob(Config.working_directory + "/predictions/*")
    prediction_ricci_no_subject_independence = None
    prediction_no_subject_independence = None
    prediction_subject_independence = []
    prediction_ricci_subject_independence = []
    for prediction_path in predictions_paths:
        identifier, ricci, number_of_model = ModelTester.get_object_data_from_path(prediction_path)
        if number_of_model is None:
            if ricci:
                prediction_ricci_no_subject_independence = extract_data_from_report(pd.read_csv(prediction_path))
            else:
                prediction_no_subject_independence = extract_data_from_report(pd.read_csv(prediction_path))
        else:
            if ricci:
                prediction_ricci_subject_independence.append(extract_data_from_report(pd.read_csv(prediction_path)))
            else:
                prediction_subject_independence.append(extract_data_from_report(pd.read_csv(prediction_path)))

    error_ricci_no_subject_independence = prediction_ricci_no_subject_independence[
        ["max_error", "distance_from_center"]] \
        .sort_values("distance_from_center", ascending=True)
    error_no_subject_independence = prediction_no_subject_independence[["max_error", "distance_from_center"]] \
        .sort_values("distance_from_center", ascending=True)

    error_subject_independence = prediction_subject_independence[0]
    for x in prediction_ricci_no_subject_independence[1:]:
        error_subject_independence = error_subject_independence.concat(x)

    error_ricci_subject_independence = prediction_ricci_subject_independence[0]
    for x in prediction_ricci_subject_independence[1:]:
        error_ricci_subject_independence = error_subject_independence.concat(x)
