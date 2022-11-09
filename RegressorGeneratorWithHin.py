import cv2
import pandas as pd
import pandas as pd
import numpy as np

import stellargraph as sg
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN, GCNSupervisedGraphClassification
from stellargraph import StellarGraph
import torch
from sklearn import model_selection
import tensorrt
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.initializers import Zeros
from keras import metrics
from matplotlib import figure
import Config
import networkx as nx

import DataUtils
import Config
import GraphGenerator
import ImageAnalizer

oracle_list = DataUtils.data_loader(Config.working_directory + "oracle_list.pickle")
"""
networkx_list = DataUtils.data_loader(Config.working_directory + "array_graph_networkx.pickle")
if Config.RegressionSetting.apply_RicciCurvature:
    networkx_list = DataUtils.apply_RicciCurvature_on_list(networkx_list)
    DataUtils.data_saver(Config.working_directory+"RiccisCurvatureGraphs.pickle", networkx_list)

pandas_oracle = pd.DataFrame.from_dict(oracle_list)
pandas_graph_list = DataUtils.networkx_list_to_pandas_list(networkx_list)
print(pandas_oracle)

stellargraph_graphs = []
for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
    stellargraph_graphs.append(StellarGraph(
        {"landmark": graph["nodes"]}, {"line": graph["edges"]}))

DataUtils.data_saver(Config.working_directory + "stellargraph_graph.pickle", stellargraph_graphs)
"""
networkx_list = DataUtils.data_loader(Config.working_directory + "RiccisCurvatureGraphs.pickle")
pandas_oracle = pd.DataFrame.from_dict(oracle_list)
pandas_graph_list = DataUtils.networkx_list_to_pandas_list(networkx_list)
print(pandas_oracle)

stellargraph_graphs = []
for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
    stellargraph_graphs.append(StellarGraph(
        {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
print(torch.version.cuda)
pandas_oracle = pd.DataFrame.from_dict(oracle_list)
print(stellargraph_graphs[0].info())

generator = PaddedGraphGenerator(graphs=stellargraph_graphs)


def create_graph_classification_model(generator_internal):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["tanh", "tanh"],
        generator=generator_internal,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="tanh")(x_out)
    predictions = Dense(units=24, activation="tanh")(predictions)
    predictions = Dense(units=1)(predictions)

    # Let's create the Keras model and prepare it for training
    output_model = Model(inputs=x_inp, outputs=predictions)
    output_model.compile(optimizer=Adam(0.005), loss='mean_squared_error', metrics=[metrics.mean_squared_error])

    return output_model


epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation
n_repeats = 5  # the number of repeats for repeated k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)




test_accs = []

train_graphs, test_graphs = model_selection.train_test_split(
    pandas_oracle["pitch"], train_size=0.7, test_size=None, random_state=20
)

train_gen = generator.flow(  # dati per l'addestramento
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=40,
    symmetric_normalization=False,
)

test_gen = generator.flow(  # dati per il test
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

model = create_graph_classification_model(generator)

history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=1, callbacks=[es],
    )# calculate performance on the test data and return along with history
model.save(Config.working_directory + "model.h5")
test_metrics = model.evaluate(test_gen, verbose=0)
test_acc = test_metrics[model.metrics_names.index("mean_squared_error")]

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))


"""
print(sg.utils.plot_history(history))

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print(model.predict(test_gen))
"""