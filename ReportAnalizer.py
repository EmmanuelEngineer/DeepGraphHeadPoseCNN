import glob

import numpy
import numpy as np

import Config
import DataUtils
import ModelTester
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error

from IPython.display import display


def plot_results(results, title):
    results.plot.bar(rot=0, title=title, figsize=(15, 8))

    plt.legend()
    plt.xlabel("Verità Effettiva")
    plt.ylabel("Errore")
    plt.savefig(Config.working_directory + "v" + str(Config.version) + "/plots"
                                                                       "/" + title + ".png")
    plt.clf()


def get_average_for_interval(dataframe, intervals, string):
    errors_in_intervals = []
    for interval in range(len(intervals)):
        errors_in_intervals.append([])

    for datapoint in dataframe[[string, string + "_error"]].to_numpy():
        for interval, (start, finish) in enumerate(intervals):
            if start <= datapoint[0] < finish:
                errors_in_intervals[interval].append(datapoint[1])
                break
    average_for_interval = []
    for interval in errors_in_intervals:
        average_for_interval.append(numpy.mean(interval))

    return average_for_interval


def get_averages(dataframe):
    intervals = []

    for start in range(-80, 80, 10):
        y = start + 10
        intervals.append((start, y))

    averages_for_pitch = get_average_for_interval(dataframe, intervals, "pitch")
    averages_for_yaw = get_average_for_interval(dataframe, intervals, "yaw")
    averages_for_roll = get_average_for_interval(dataframe, intervals, "roll")

    return pd.DataFrame({"pitch": averages_for_pitch,
                         "yaw": averages_for_yaw,
                         "roll": averages_for_roll}, index=intervals)


def extract_errors_from_data(report):
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


def calculate_accuracies(dataframe, title):
    pitch_acc = accuracy_score(dataframe["pitch"], dataframe["predicted_pitch"])
    roll_acc = accuracy_score(dataframe["roll"], dataframe["predicted_roll"])
    yaw_acc = accuracy_score(dataframe["yaw"], dataframe["predicted_yaw"])
    mean_error = mean_absolute_error(dataframe[["pitch", "yaw", "roll"]],
                                     dataframe[["pitch_error", "yaw_error", "roll_error"]])
    print("accuracy of %s , pitch %f, yaw %f, roll %f, MAE %f" % title, pitch_acc, roll_acc, yaw_acc, mean_error)


if __name__ == "__main__":
    predictions_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/predictions/*")
    edge_type_index = []
    report_container = []

    for prediction_path in predictions_paths:
        identifier, edge_type, number_of_model = ModelTester.get_object_data_from_path(prediction_path)
        print(prediction_path)

        if number_of_model is None:
            prediction_data = extract_errors_from_data(pd.read_csv(prediction_path))
            plot_results(get_averages(prediction_data), identifier)
            calculate_accuracies(prediction_data,identifier)

        else:
            if edge_type not in edge_type_index:
                edge_type_index.append(edge_type)
                report_container.append([])
            idx = edge_type_index.index(edge_type)
            report_container[idx].append(extract_errors_from_data(pd.read_csv(prediction_path)))

    for idx, x in enumerate(edge_type_index):
        results = get_averages(pd.concat(report_container[idx]))
        plot_results(results, x + "_leave_one_out")
        calculate_accuracies(results, x + "_leave_one_out")
