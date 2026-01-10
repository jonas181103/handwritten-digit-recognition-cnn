import matplotlib.pyplot as plt
import numpy
import src.NeuralNetwork as NeuralNetwork
import scipy.special  # Sigmoid-Funktion


def initiate_neuralnetwork(input_nodes:int, output_nodes:int, learningrate:float = 0.3):
    """Berechne die Anzahl der Nodes für das neuronale Netzwerk anhand von Input und Output und
    erstelle ein NN-Objekt.
    """
    input_nodes = input_nodes
    # Anzahl der Hidden Nodes wird anhand des Richtwertes für CNNs dynamisch bestimmt
    hidden_nodes = int((input_nodes+output_nodes)//(3/2))
    output_nodes = output_nodes
    learning_rate = learningrate
    neural_network = NeuralNetwork.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    return neural_network

def train_neuralnetwork(epochs:int = 5):
    dataset_file = "data/raw/mnist_data/Testdaten/mnist_train_100.csv"
    training_data_list = read_dataset(dataset_file)
    print(f"Training neural network with dataset \"{dataset_file}\" in {epochs} epochs")
    for e in range(epochs):
        # Alle Zeilen im Dataset sind ein record und repräsentieren eine Zahl
        for record in training_data_list:
            pixel_values = record.split(',')
            # Konvertiere Pixeldaten 0-255 zu float 0-1
            inputs = (numpy.asarray(pixel_values[1:], float) / 255.0 * 0.99) + 0.01
            # Ziel festlegen: alle Werte des Outputs auf 0.01 außer der gewünschte Wert
            targets = numpy.zeros(n.onodes) + 0.01
            targets[int(pixel_values[0])] = 0.99
            n.train(inputs, targets)
            pass
        print(f"Epoch {e + 1} done")
        pass
    print("Training finished.\n")

def ask_neuralnetwork(pixel_values, display:bool = True, print_output:bool = True):
    # Anzeige der Zahl mit Matplotlib
    if display:
        image_array = numpy.asarray(pixel_values, dtype="float").reshape((28, 28))
        plt.imshow(image_array, cmap="Greys", interpolation="None")
        plt.show()
    # Query and das CNN
    outputs = n.query((numpy.asarray(pixel_values, dtype="float") / 255.0 * 0.99) + 0.01)
    # Ausgabe der Werte
    if print_output:
        counter = 0
        for value in outputs:
            print(f"Number: {counter} with certainty {value[0]:%}")
            counter += 1

    # Das Label zurückgeben (Was das CNN denkt, der Wert ist)
    return numpy.argmax(outputs)

def test_neuralnetwork(test_dataset):
    print(f"Testing neural network...")
    # Eine Scorecard zur Bestimmung der Leistung
    scorecard = []

    # Durch alle records im Dataset gehen und testen
    for record in test_dataset:
        all_values = record.split(',')
        # Der erste Wert ist die Zahl die wir suchen
        correct_label = int(all_values[0])
        label = ask_neuralnetwork(all_values[1:], display=False, print_output=False)
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass

        pass
    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print(f"Performance = {(scorecard_array.sum() / scorecard_array.size):.2%}\n")

def read_dataset(file_path:str):
    """Auslesen des Datensatzes mit Fehlerbehandlung. Erwartet wird eine CSV-Datei,
    die die Zahl und die Werte beinhaltet.
    """
    try:
        data_file = open(file_path, "r")
        data_list = data_file.readlines()
        data_file.close()
    except FileNotFoundError as e:
        print(f"Error: {e.strerror} for path: {e.filename}. \nMake sure the file exists and is accessible.")
        exit(1)
    except IOError as e:
        print(f"Unexpected Error: {e.strerror} for path: {e.filename}.")
        exit(1)
    return data_list

# Erzeuge neuronales Netzwerkobjekt n mit der Anzahl der Pixel als Anzahl der Nodes im Input Layer.
# Wir setzen voraus, dass alle Bilder die gleiche Größe haben und eine Zahl von 0 bis 9 gefunden werden soll
n_pixel = 784
n_output = 10
n = initiate_neuralnetwork(n_pixel, n_output)

# Neuronales Netzwerk trainieren
train_neuralnetwork()

# Neuronales Netzwerk testen
# Testdatensatz einlesen (Besteht aus einer CSV mit einer Zahl und einem Bild für die Zahl pro Zeile)
test_list = read_dataset("data/raw/mnist_data/Testdaten/mnist_test_10.csv")
test_neuralnetwork(test_list)
