import Config
import DataUtils
import ImageAnalizer
import stellargraph as sg
import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.saving.save import load_model


images_paths = ImageAnalizer.import_images_paths(Config.Test.image_dataset)
model = load_model("Dataset/Test/model.h5")

# TODO Stellargraph conversion of input data
