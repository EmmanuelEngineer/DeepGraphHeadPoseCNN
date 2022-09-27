Link del tutorial ufficiale StellarGraph
https://stellargraph.readthedocs.io/en/stable/demos/graph-classification/dgcnn-graph-classification.html

predictions = Dense(units=1)(x_out)

Nel giro di pochi giorni puoi completare i seguenti tre passi.

Step 1: Scarica il seguente dataset https://drive.google.com/file/d/1PJtn5UwuUazrrw9OZT7H_u_DYAx9FfYk/view?usp=sharing
La nomenclatura di ciascuna immagine riguarda, in ordine, gli assi di pitch, yaw, roll.

Step 2: Estrazione dei landmarks, usa la libreria MediaPipe, qui un esempio: https://www.analyticsvidhya.com/blog/2022/03/facial-landmarks-detection-using-mediapipe-library/

Step 3: Creazione di un grafo, aiutati con la libreria: https://networkx.org/documentation/stable/tutorial.html
Seleziona i landmarks che NON appartengono alle parti mobili, occhi, sopracciglia, bocca, zigomi ecc. Unisci quelli che restano in questo modo:
ciascun landmark (che sarà poi il tuo vertice) viene collegato ai 5 più vicini. 