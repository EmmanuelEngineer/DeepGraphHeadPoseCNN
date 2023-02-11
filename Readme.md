DeepGraphHeadPoseCNN: A tool to analize different metrics for graph based head pose estima
# Welcome to DeepGraphHeadPoseCNN!

This is a bachelor degree thesis project. The **objective** of the project was to test different metrics that can be useful for graph based head pose estimation using facial landmarks as nodes.
Motivation? Landmarks are a tool already used for different application in multimodal computer vision and could be useful to find a away to use that data already present for other applications.



# Files
Following: a brief description of the most important files that are part of the project.

## ImageAnalizer
To test and evaluate the pose estimator we need data. ImageAnalizer uses Google's library **Mediapipe** to extract the landmarks from an image dataset. The dataset used for the project was a pre-processed version of **BIWI** dataset where the poses were already converted in rgb images and the true pose was written in the name of the file.
During the analysis, the landmarks coordinates are **normalized**  to eliminate disturbance from faces that are outside the center of the frame.
Some landmarks are filtered to data-diminish noise for the neural network.
## GraphGenerator
After obtained the landmarks was needed to generate the graphs, using **scipy** libraries to calculate the distance between nodes. The algorithm applied was the k-nearest neighbor with k as 5.
>**NOTE:** i decided to switch between metrics creating a function for each metric instead of using only a function and a switch or series of "if"for efficiency purpose. 
I didn't want to activate the same chain of "if" thousands of time for the same graph.

![Visualizzation of what Image Genarator and Graph generator did to the data.](https://drive.google.com/file/d/1g71UvlDXNQ8ZNs572-co5Y5WfALTyMhL/view?usp=share_link)

## ApplyRiccisCurvature

Same thing as Graph generator but it applies Ricci's Curvature.

## RegressorGenarator
This file generates and trains the graph neural network based on chosen metrics and type of split.
The neural network is based on a **CNN** built for protein classification [(DeepGraphCNN)](https://stellargraph.readthedocs.io/en/v1.2.1/demos/graph-classification/dgcnn-graph-classification.html). It is expanded in number of units to accept the bigger graphs (is not optimized), some activation function were changed to keep useful data and to change from classification to regression. 
Depending on the Config file setting, it will use a 70 10 20 train-validation-test split of the dataset or will do a leave-one-subject-out split creating a neural network for each subject used as a test.

## ModelTester

This file will test every neural network and build a report.
## ReportAnalyzer
Using the reports will generate graphs based on the split and metric type.




> Written with [StackEdit](https://stackedit.io/).tion
