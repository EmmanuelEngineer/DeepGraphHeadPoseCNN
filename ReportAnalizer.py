import glob

import numpy

import Config
import ModelTester
import pandas as pd
import matplotlib.pyplot as plt


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


def extract_errors_from_data(dataframe):
    max_list = dataframe[["pitch", "yaw", "roll"]].abs().max(axis=1)
    dataframe["distance_from_center"] = max_list.values

    error_pitch_eval = "pitch_error = dataframe.pitch - dataframe.predicted_pitch"
    error_yaw_eval = "yaw_error = dataframe.yaw - dataframe.predicted_yaw"
    error_roll_eval = "roll_error = dataframe.roll - dataframe.predicted_roll"

    dataframe = pd.eval(error_pitch_eval, target=dataframe)
    dataframe = pd.eval(error_yaw_eval, target=dataframe)
    dataframe = pd.eval(error_roll_eval, target=dataframe)

    dataframe["pitch_error"] = dataframe["pitch_error"].abs()
    dataframe["yaw_error"] = dataframe["yaw_error"].abs()
    dataframe["roll_error"] = dataframe["roll_error"].abs()

    max_list = dataframe[["pitch_error", "yaw_error", "roll_error"]].abs().max(axis=1)
    dataframe["max_error"] = max_list.values
    max_list = dataframe[["pitch_error", "yaw_error", "roll_error"]].abs().idxmax(axis=1)
    dataframe["max_error_axis"] = max_list.values

    dataframe[0] = None

    return dataframe


def calculate_accuracies(dataframe, title):
    pitch_acc = dataframe["pitch_error"].mean()
    roll_acc = dataframe["roll_error"].mean()
    yaw_acc = dataframe["yaw_error"].mean()
    mean_error = (pitch_acc+ roll_acc +yaw_acc)/3
    print("accuracy of %s , pitch %f, yaw %f, roll %f, MAE %f" % (title, pitch_acc, roll_acc, yaw_acc, mean_error))


if __name__ == "__main__":
    predictions_paths = glob.glob(Config.working_directory + "v" + str(Config.version) + "/predictions/*")
    edge_type_index = []
    report_container = []

    for prediction_path in predictions_paths:
        identifier, edge_type, number_of_model = ModelTester.get_object_data_from_path(prediction_path)
        print(prediction_path)

        if number_of_model is None:
            prediction_data = extract_errors_from_data(pd.read_csv(prediction_path))
            plot_results(get_averages(prediction_data), edge_type + "test_split")
            calculate_accuracies(prediction_data, identifier)

        else:
            if edge_type not in edge_type_index:
                edge_type_index.append(edge_type)
                report_container.append([])
            idx = edge_type_index.index(edge_type)
            report_container[idx].append(extract_errors_from_data(pd.read_csv(prediction_path)))

    for idx, x in enumerate(edge_type_index):
        report = pd.concat(report_container[idx])
        averages = get_averages(report)
        plot_results(averages, x + "_loso")
        calculate_accuracies(report, x + "_loso")
