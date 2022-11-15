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
    for model_path in enumerate(model_paths):
        if "ricci" in re.findall()
        model = load_model(model_path)
        history = None
        for history_path in histories_paths:
            re.findall(regex, history_path)
