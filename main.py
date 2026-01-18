import matplotlib.pyplot as plt
import numpy
import src.NeuralNetwork as NeuralNetwork
import src.REPatternMatcher as repm
import os
import pandas as pd
#import scipy.special  # Sigmoid-Funktion

__author__ = "Jonas Ott, Simon Wameling, ..."

# Globale Variablen für Pfade und Dateien, die häufiger verwendet werden
MODELS_DIR = "models"
VISUALIZATION_DIR = "data/visualisation"
TRAIN_DATASET = "data/raw/mnist_data/testdaten/mnist_train_100.csv"
TEST_DATASET = "data/raw/mnist_data/testdaten/mnist_test_10.csv"

# Lernfortschritt als Diagramm anzeigen
def create_plot(accuracy_dictionary):
    if not accuracy_dictionary:
        print("Keine Daten zum Visualisieren vorhanden.")
        return

    # Daten aus dem Dictionary extrahieren und nach Epochen sortieren
    epochs_sorted = sorted(accuracy_dictionary.keys())
    performances = [accuracy_dictionary[a] for a in epochs_sorted]

    # Layout des Diagramms
    plt.figure(figsize=(10, 6))

    # 3. Die Kurve zeichnen mit 'epoch' auf der X-Achse, 'performance' auf der Y-Achse
    plt.plot(epochs_sorted, performances,
             marker='o',  # Punkte an den Datenwerten
             linestyle='-',  # Durchgezogene Linie
             color='#2ecc71',  # Farbe Grün
             linewidth=2,
             label='Accuracy')

    # Titel und Beschriftungen
    plt.title('Lernfortschritt des Neuronalen Netzes', fontsize=14)
    plt.xlabel('Epoche', fontsize=12)
    plt.ylabel('Genauigkeit (Performance)', fontsize=12)
    # Gitter im Hintergrund für eine bessere Lesbarkeit
    plt.grid(True, linestyle='--', alpha=0.6)
    # Legende anzeigen
    plt.legend()
    # Als Bild speichern
    try:
        plt.savefig(VISUALIZATION_DIR + "accuracy_trend.png")
    except FileNotFoundError as e:
        print(f"Error: {e.strerror} for path: {e.filename}. \n")
    except IOError as e:
        print(f"Unexpected Error: {e.strerror} for path: {e.filename}.")
        exit(1)
    plt.show()


def initiate_neuralnetwork(input_nodes:int, output_nodes:int, learning_rate:float = 0.3):
    """Berechne die Anzahl der Nodes für das neuronale Netzwerk anhand von Input und Output und
    erstelle ein NN-Objekt.
    """
    input_nodes = input_nodes
    # Anzahl der Hidden Nodes wird anhand des Richtwertes für CNNs dynamisch bestimmt
    hidden_nodes = int((input_nodes+output_nodes)//(3/2))
    output_nodes = output_nodes
    learning_rate = learning_rate
    neural_network = NeuralNetwork.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    return neural_network

def train_neuralnetwork(epochs:int = 5, print_output:bool = True):
    dataset_file = TRAIN_DATASET
    training_data_list = read_dataset(dataset_file)
    if print_output:
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
        if print_output:
            print(f"Epoch {e + 1} done")
        pass
    if print_output:
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
            print(f"Number: {counter} with certainty {value[0]:.2%}")
            counter += 1

    # Das Label zurückgeben (Was das CNN denkt, der Wert ist)
    return numpy.argmax(outputs)

def test_neuralnetwork(test_dataset, print_output:bool = True):
    if print_output:
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
    performance = numpy.mean(scorecard_array)
    if print_output:
        print(f"Performance = {performance:.2%}\n")
    return performance

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

def list_models():
    """Liefert alle .npz Dateien im aktuellen Verzeichnis als Rückgabewert."""
    try:
        # Prüfen, ob der Ordner existiert
        if not os.path.exists(MODELS_DIR):
            print(f"Directory '{MODELS_DIR}' does not exist.")
            return False

        # Dateien im Ordner scannen
        files = [f for f in os.listdir(MODELS_DIR) if
                 os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith('.npz')]

        # Wenn es keine Models gibt, können wir nichts laden
        if not files:
            return False

        return files
    except OSError as e:
        print(f"Error: {e.strerror}.")
        return False


def print_help():
    """Zeigt die Hilfe an"""
    help_data = {
        "Command": ["Train", "Test", "Save", "Load", "End", "Help"],
        "Description": [
            "Trains the network",
            "Tests the network with test data",
            "Saves the neural networks' weights in 'models/'",
            "Loads the weights in 'models/'",
            "Exits the program",
            "Shows the help"
        ]
    }

    # Ein Pandas DataFrame zur Darstellung der Tabelle
    df = pd.DataFrame(help_data)
    print("--- Available Commands ---" + "-"* (61-26))
    # index=False entfernt die Zeilennummern (0, 1, 2...)
    print(df.to_string(index=False, justify='left',
                       formatters={'Command': lambda x: f"{x:<10}", 'Description': lambda x: f"{x:<50}"}))
    # Trennlinie genau passend zur Tabellenbreite (10 + 1 + 50 = 61)
    print("-" * 61 + "\n")

# 1. INIT
# Erzeuge neuronales Netzwerkobjekt n mit der Anzahl der Pixel als Anzahl der Nodes im Input Layer.
# Wir setzen voraus, dass alle Bilder die gleiche Größe haben und eine Zahl von 0 bis 9 gefunden werden soll
n_pixel = 784
n_output = 10
n = initiate_neuralnetwork(n_pixel, n_output)

# 2. MAIN LOOP
while True:
    try:
        user_input = input(f"What do you want to do? [Train|Test|Save|Load|End|Help]\n")
        match repm.REqual(user_input):
            case r'^[T|t]rain':
                # Benutzerabfrage
                try:
                    epochs = int(input("How many epochs do you want to train?\n"))
                    visualize = bool(input("Do you want visualization? [True/False]\n]"))
                except ValueError as e:
                    print(f"Error in user input. Try again!")
                    break

                # Da beim Visualisieren ein anderer Ansatz gewählt wird hier unterscheiden
                if visualize:
                    accuracy_data = {}
                    # Trainieren mit jeweils einer Epoche für accuracy Daten
                    for epoch in range(epochs):
                        train_neuralnetwork(1, False)
                        accuracy = test_neuralnetwork(read_dataset(TEST_DATASET), False)
                        accuracy_data[epoch] = accuracy

                        print(f"Epoch {epoch}: Accuracy = {accuracy:.5f}")

                    create_plot(accuracy_data)

                    # Visualisieren
                    for key, value in accuracy_data.items():
                        print(f"Epoch: {key}, Accuracy: {value}")
                else:
                    # Neuronales Netzwerk trainieren
                    train_neuralnetwork(epochs)
            case r'^[T|t]est':
                # Neuronales Netzwerk testen
                # Testdatensatz einlesen (Besteht aus einer CSV mit einer Zahl und einem Bild für die Zahl pro Zeile)
                test_list = read_dataset(TEST_DATASET)
                test_neuralnetwork(test_list)
            case r'^[E|e]nd|^[E|e]xit':
                print(f"Exiting...")
                exit(0)
            case r'^[S|s]ave':
                models = list_models()
                if models:
                    print(f"Found {len(models)} model(s):")
                    for model in models:
                        print(f"  -> {model}")
                filename = input("Please type in the file name (existing files will be overwritten!): ").strip()
                if not filename:
                    filename = "weights.npz"
                n.save_weights(filename)
            case r'^[L|l]oad':
                # Erst auflisten. Wenn nichts da ist, brauchen wir auch nichts abzufragen.
                models = list_models()
                if models:
                    print(f"Found {len(models)} model(s):")
                    for model in models:
                        print(f"  -> {model}")
                    filename = input("Which model do you want to load? [file name]: ").strip()
                    # Standardwert nutzen, wenn Enter gedrückt wurde
                    if not filename:
                        filename = "weights.npz"
                    # Automatisch .npz anhängen, falls der User es vergessen hat
                    if not filename.endswith(".npz"):
                        filename += ".npz"
                    n.load_weights(filename)
                else:
                    print(f"No models found.")
            case r'^[H|h]elp':
                print_help()
            case _:
                print(f"Command not recognized: {user_input}. Type 'Help' for help.")
    except KeyboardInterrupt:
        print(f"User interrupted")
        exit(0)