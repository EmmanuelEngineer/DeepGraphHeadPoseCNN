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
stellargraph_graphs = DataUtils.data_loader(Config.working_directory + "stellargraph_graph.pickle")
print(torch.version.cuda)
pandas_oracle = pd.DataFrame.from_dict(oracle_list)
print(stellargraph_graphs[0].info())


generator = PaddedGraphGenerator(graphs=stellargraph_graphs)


def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.005), loss=binary_crossentropy, metrics=["acc"])

    return model


epochs = 200  # maximum number of training epochs
folds = 10  # the number of folds for k-fold cross validation
n_repeats = 5  # the number of repeats for repeated k-fold cross validation

es = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=25, restore_best_weights=True
)


def train_fold(model, train_gen, test_gen, es, epochs):
    history = model.fit(
        train_gen, epochs=epochs, validation_data=test_gen, verbose=0, callbacks=[es],
    )
    # calculate performance on the test data and return along with history
    test_metrics = model.evaluate(test_gen, verbose=0)
    test_acc = test_metrics[model.metrics_names.index("acc")]

    return history, test_acc


def get_generators(train_index, test_index, graph_labels, batch_size):
    train_gen = generator.flow(
        train_index, targets=graph_labels.iloc[train_index].values, batch_size=batch_size
    )
    test_gen = generator.flow(
        test_index, targets=graph_labels.iloc[test_index].values, batch_size=batch_size
    )

    return train_gen, test_gen


test_accs = []

stratified_folds = model_selection.RepeatedStratifiedKFold(
    n_splits=folds, n_repeats=n_repeats
).split(pandas_oracle["pitch"], pandas_oracle["pitch"])

for i, (train_index, test_index) in enumerate(stratified_folds):
    print(f"Training and evaluating on fold {i + 1} out of {folds * n_repeats}...")
    train_gen, test_gen = get_generators(
        train_index, test_index, pandas_oracle["pitch"], batch_size=30
    )

    model = create_graph_classification_model(generator)

    history, acc = train_fold(model, train_gen, test_gen, es, epochs)

    test_accs.append(acc)

print(
    f"Accuracy over all folds mean: {np.mean(test_accs)*100:.3}% and std: {np.std(test_accs)*100:.2}%"
)



"""
print(sg.utils.plot_history(history))

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print(model.predict(test_gen))
"""
model.save(Config.working_directory + "model.h5")

images_paths = ImageAnalizer.import_images_paths(Config.Test.image_dataset)
point_array = ImageAnalizer.landmark_extraction(images_paths)

print(images_paths)

for index, image_path in enumerate(images_paths):
    oracle_list = ImageAnalizer.extract_oracle([image_path])
    image = cv2.imread(image_path)
    image = image[:, :, ::-1].copy()
    edge_list = GraphGenerator.edge_list_generator(point_array[0])
    image_rows, image_cols, _ = image.shape
    if Config.Extraction.scale_landmarks:
        image_rows = 1000
        image_cols = 1000
    image = np.zeros((image_rows, image_cols, 3), dtype="uint8")
    DataUtils.draw_edges(image,
                         point_array[index],
                         [(x[0], x[1]) for x in edge_list])
    DataUtils.draw_landmarks(image, point_array[index], original=True)
    graph = nx.Graph()
    nodes_to_graph = []
    for idn, x in enumerate(point_array[0]):
        nodes_to_graph.append((idn, x))  # Because networkx needs the node index to work
    graph.add_nodes_from(nodes_to_graph)
    graph.add_edges_from(edge_list)
    fig = plt.figure(figsize=[10, 10], dpi=300)
    plt.title("Resultant Image")
    plt.axis('off')
    plt.imshow(image)
    plt.show()

    pandas_graph_list = DataUtils.networkx_list_to_pandas_list([graph])
    pandas_oracle = pd.DataFrame.from_dict(oracle_list)
    pandas_graph_list[0]["nodes"].to_csv("nodes.csv")
    pandas_graph_list[0]["edges"].to_csv("edges.csv")

    stellargraph_graphs = []
    for graph in pandas_graph_list:  # Conversion to stellargraph Graphs
        stellargraph_graphs.append(StellarGraph(
            {"landmark": graph["nodes"]}, {"line": graph["edges"]}))
    test_generator = PaddedGraphGenerator(graphs=stellargraph_graphs)
    obj = test_generator.flow(stellargraph_graphs)
    predict = model.predict(obj)
    print("predicted", predict, "expected", oracle_list[0])
