import glob

import numpy as np

import Config
import DataUtils
import ModelTester
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display


def extract_data_from_report(report):
    max_list = report[["pitch", "yaw", "roll"]].abs().max(axis=1)
    report["distance_from_center"] = max_list.values

    error_pitch_eval = "pitch_error = report.pitch - report.predicted_pitch"
    error_yaw_eval = "yaw_error = report.yaw - report.predicted_yaw"
    error_roll_eval = "roll_error = report.roll - report.predicted_roll"

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
    report[0] = None

    return report


if __name__ == "__main__":
    predictions_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/predictions/*")
    prediction_no_subject_independence = []
    prediction_subject_independence = []
    prediction_ricci_subject_independence = []
    prediction_ricci_no_subject_independence = []

    for prediction_path in predictions_paths:
        identifier, ricci, number_of_model = ModelTester.get_object_data_from_path(prediction_path)
        print(prediction_path)
        if number_of_model != 6:
            continue

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

    plt.figure(figsize=(15, 8))
    if len(prediction_ricci_no_subject_independence) > 0:
        error_r_n_s = prediction_ricci_no_subject_independence[
            ["max_error", "distance_from_center"]] \
            .sort_values("distance_from_center", ascending=True)
        plt.plot(error_r_n_s["distance_from_center"],
                 error_r_n_s["max_error"], label="ricci no sbi", linewidth=0.5)

    if len(prediction_no_subject_independence) > 0:
        error_n_s = prediction_no_subject_independence[["max_error", "distance_from_center"]] \
            .sort_values("distance_from_center", ascending=True)
        plt.plot(error_n_s["distance_from_center"], error_n_s["max_error"],
                 label="no ricci no sbi", linewidth=0.5)

    if len(prediction_subject_independence) > 0:
        error_s = prediction_subject_independence[0][["max_error", "distance_from_center"]]
        for x in prediction_subject_independence[1:]:
            error_s = error_s.append(x[["max_error", "distance_from_center"]], ignore_index=True)
        error_s = error_s.sort_values("distance_from_center", ascending=True)
        plt.plot(error_s["distance_from_center"], error_s["max_error"],
                 label="no ricci sbi", linewidth=0.5)

    if len(prediction_ricci_subject_independence) > 0:
        error_r_s = prediction_ricci_subject_independence[0][["max_error", "distance_from_center"]]
        for x in prediction_ricci_subject_independence[1:]:
            error_r_s = error_r_s.append(x[["max_error", "distance_from_center"]], ignore_index=True)
        error_r_s = error_r_s.sort_values("distance_from_center", ascending=True)
        plt.plot(error_r_s["distance_from_center"], error_r_s["max_error"],
                 label="ricci sbi", linewidth=0.5)

    plt.legend()
    plt.xlabel("Distanza dal centro")
    plt.ylabel("Errore in gradi")
    plt.show()
