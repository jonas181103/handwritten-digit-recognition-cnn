""" Dieses Projekt dient der automatisierten Erkennung
und Klassifikation handschriftlicher Ziffern (0-9).
Es demonstriert die Anwendung von Deep Learning
und speziell eines Convolutional Neural Network (CNN).
Inspiration für REPatternMatcher:
https://discuss.python.org/t/structural-pattern-matching-should-permit-regex-string-matches/22700/9
"""

# Standard-Bibliotheken importieren
import os
import sys
# Third-Party-Bibliotheken importieren
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Eigene Module importieren
import src.neural_network as nn
import src.re_pattern_matcher as repm
import src.image_loader as il

__author__ = "Jonas Ott, Simon Wameling, ..."

# Globale Variablen für Pfade und Dateien und Verzeichnisse, die häufiger verwendet werden
MODELS_DIR = "models"
VISUALIZATION_DIR = "data/visualisation/"
PIC_DIR = "data/pictures/"
TRAIN_DATASET = "data/raw/mnist_data/echtdaten/mnist_test.csv"
TEST_DATASET = "data/raw/mnist_data/testdaten/mnist_train_100.csv"
N_PIXEL = 784
N_OUTPUT = 10


def create_plot(accuracy_dictionary):
    """
    Lernfortschritt als Diagramm anzeigen.

    :param accuracy_dictionary: Dict, in dem die Accuracy mit Epoche als Key gespeichert ist
    :return: None
    """
    if not accuracy_dictionary:
        print("Keine Daten zum Visualisieren vorhanden.")
        return

    # Daten aus dem Dictionary extrahieren und nach Epochen sortieren
    epochs_sorted = sorted(accuracy_dictionary.keys())
    performances = [accuracy_dictionary[a] for a in epochs_sorted]

    # Layout des Diagramms
    plt.figure(figsize=(10, 6))

    # Die Kurve zeichnen mit 'epoch' auf der x-Achse, 'performance' auf der y-Achse
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
        plt.savefig(os.path.join(VISUALIZATION_DIR + "accuracy_trend.png"))
    except FileNotFoundError as e_create_plot:
        print(f"Error: {e_create_plot.strerror} for path: {e_create_plot.filename}. \n")
    except IOError as e_create_plot:
        print(f"Unexpected Error: {e_create_plot.strerror} for path: {e_create_plot.filename}.")
    plt.show()


def initiate_neuralnetwork(p_input_nodes: int, p_output_nodes: int, p_learning_rate: float = 0.3):
    """ Berechnet die Anzahl derv Hidden Nodes für das neuronale Netzwerk
     anhand von Input und Output und erstellt ein NN-Objekt.

    :param p_input_nodes: Anzahl der Input-Nodes des neuronalen Netzwerks
    :param p_output_nodes: Anzahl der Output-Nodes des neuronalen Netzwerks
    :param p_learning_rate: stellt die Lernrate des NN-Objektes ein
    :return: das neu erstellte NN-Objekt
    """
    input_nodes = p_input_nodes
    # Anzahl der Hidden Nodes wird anhand des Richtwertes für CNNs dynamisch bestimmt
    hidden_nodes = int((input_nodes + p_output_nodes) // (3 / 2))
    output_nodes = p_output_nodes
    learning_rate = p_learning_rate
    neural_network = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    return neural_network


def train_neuralnetwork(p_neuralnetwork, dataset_file=TRAIN_DATASET,
                        epochs: int = 5, print_output: bool = True):
    """
    Funktion zum Vorbereiten vom Trainieren des neuronalen Netzwerks mittels train() im NN-Objekt.

    :param p_neuralnetwork: NN-Objekt, dass trainiert werden soll
    :param dataset_file: Datei, die das Dataset enthält (csv)
    :param epochs: für wie viele Epochen trainiert wird
    :param print_output: soll Rückmeldung per print gegeben werden?
    :return:
    """
    training_data_list = read_dataset(dataset_file)
    if print_output:
        print(f"Training neural network with dataset \"{dataset_file}\" in {epochs} epochs")
    for i in range(epochs):
        # Alle Zeilen im Dataset sind ein record und repräsentieren eine Zahl
        for record in training_data_list:
            pixel_values = record.split(',')
            # Konvertiere Pixeldaten 0-255 zu float 0-1
            inputs = (np.asarray(pixel_values[1:], float) / 255.0 * 0.99) + 0.01
            # Ziel festlegen: alle Werte des Outputs auf 0.01 außer der gewünschte Wert
            targets = np.zeros(my_nn.onodes) + 0.01
            targets[int(pixel_values[0])] = 0.99
            # Neurales Netzwerk trainieren
            p_neuralnetwork.train(inputs, targets)
        if print_output:
            print(f"Epoch {i + 1} done")
    if print_output:
        print("Training finished.\n")


def list_files(directory: str, endswith = ""):
    """
    Ermittelt alle Dateien mit bestimmter Endung im gegebenen Verzeichnis.

    :param directory: Verzeichnis, das durchsucht werden soll
    :param endswith: Endung, die gesucht werden soll
    :return: Liste mit gefundenen Dateien oder False (Fehler oder nichts dort)
    """
    try:
        # Prüfen, ob der Ordner existiert
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return False

        # Dateien im Ordner scannen
        files = [f for f in os.listdir(directory) if
                 os.path.isfile(os.path.join(directory, f)) and f.endswith(endswith)]

        # Prüfen, ob es Dateien im Ordner gibt
        if not files:
            return False

        return files
    except OSError as e_list_files:
        print(f"Error: {e_list_files.strerror}.")
        return False


def print_dataset(dataset):
    """
    Ausgabe aller Bilder im Datensatz

    :param dataset:
    :return:
    """
    # Durch das Dataset iterieren
    for record in dataset:
        # Werte auslesen und den ersten Wert (Zahl, die das Bild darstellen soll) verwerfen
        pixel_values = record.split(',')
        pixel_values = pixel_values[1:]
        # Bilder ausgeben
        print_image(pixel_values)


def print_image(pixel_values):
    """
    Anzeige der Pixelwerte mit Matplotlib als Plot mit 28 x 28 Pixel

    :param pixel_values: Liste von 784 Pixelwerten (0-255), die das Eingabebild repräsentieren
    :return: None
    """
    image_array = np.asarray(pixel_values, dtype="float").reshape((28, 28))
    plt.imshow(image_array, cmap="Greys", interpolation="None")
    plt.show()


def ask_neuralnetwork(p_neuralnetwork, pixel_values: list, min_confidence: float = 0,
                      display: bool = False, print_output: bool = True):
    """
    Befragt das neuronale Netzwerk, um die eingegebenen Pixelwerte zu klassifizieren.

    :param p_neuralnetwork: NN-Objekt, dass befragt werden soll
    :param pixel_values: Liste von 784 Pixelwerten (0-255), die das Eingabebild repräsentieren
    :param min_confidence: minimaler Wert (0.0 bis 1.0), um das ermittelte Ergebnis zu akzeptieren
    :param display: ob eine Visualisierung des Lernfortschritts erstellt werden soll
    :param print_output: ermittelten Wahrscheinlichkeiten für jede Ziffer in der Konsole ausgegeben
    :return: erkannte Ziffer als Integer (0-9) oder -1, falls die Konfidenz zu niedrig war
    """
    cnn_guess = {"number": -1, "confidence":0}
    # Anzeige der Zahl mit Matplotlib
    if display:
        print_image(pixel_values)
    # Query an das CNN
    outputs = p_neuralnetwork.query((np.asarray(pixel_values, dtype="float") / 255.0 * 0.99) + 0.01)
    # Ausgabe der Werte
    counter = 0
    highest_value = np.float64(0)
    for value in outputs:
        certainty = value[0]
        if print_output:
            print(f"Number: {counter} with certainty {certainty:.2%}")
        if certainty > highest_value:
            cnn_guess = {"number": counter, "confidence": certainty}
            highest_value = certainty
        counter += 1
    if cnn_guess["confidence"] < min_confidence:
        cnn_guess = {"number": -1}

    # Das Label zurückgeben (Was das CNN denkt, der Wert ist)
    return cnn_guess["number"]


def test_neuralnetwork(p_neuralnetwork, test_dataset,
                       print_output: bool = True, plot: bool = False):
    """
    Testet das NN mit einem vorgegebenen Datensatz, um zu bestimmen, wie gut das NN ist.

    :param test_dataset: der Datensatz (nicht die Datei!) als Liste
    :param print_output: ob der CLI output ausgegeben werden soll
    :param plot: ob der Plot gezeichnet werden soll
    :return: Performance als float von 0 (schlechteste) bis 1 (beste)
    """
    if not test_dataset:
        return False
    if print_output:
        print("Testing neural network...")
    # Eine Scorecard zur Bestimmung der Leistung
    scorecard = []

    # Durch alle records im Dataset gehen und testen
    for record in test_dataset:
        all_values = record.split(',')
        # Der erste Wert ist die Zahl, die wir suchen
        correct_label = int(all_values[0])
        label = ask_neuralnetwork(p_neuralnetwork ,all_values[1:], display=plot, print_output=False)
        # append correct or incorrect to list
        if label == correct_label:
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
    # calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(scorecard)
    performance = np.mean(scorecard_array)
    if print_output:
        print(f"Performance = {performance:.2%}\n")
    return performance


def read_dataset(file_path: str):
    """
    Auslesen des Datensatzes mit Fehlerbehandlung. Erwartet wird eine CSV-Datei,
    die die Zahl und die Werte beinhaltet.

    :param file_path: Pfad, in dem sich der Datensatz als CSV-Datei befindet
    :return: Datensatz als Liste, oder False bei einem Fehler
    """
    try:
        with open(file_path, "r", encoding="utf-8") as data_file:
            # Jede Zeile ist ein Bild (weiß auf schwarz) mit der Zahl als ersten Eintrag
            return data_file.readlines()
    except FileNotFoundError as e_read_dataset:
        print(f"Error: {e_read_dataset.strerror} for path: "
              f"{e_read_dataset.filename}. \nMake sure the file exists and is accessible.")
        return False
    except IOError as e_read_dataset:
        print(f"Unexpected Error: {e_read_dataset.strerror} for path: {e_read_dataset.filename}.")
        return False


def print_help():
    """
    Zeigt die Hilfe an

    :return: None
    """
    help_data = {
        "Command": ["Train", "Test", "Ask", "Save", "Load", "Delete", "Reset", "End", "Help"],
        "Description": [
            "Trains the network",
            "Tests the network with test data",
            "Asks the neural network to analyze an image",
            "Saves the neural networks' weights in 'models/'",
            "Loads the weights in 'models/'",
            "Deletes the specified weights in 'models/'",
            "Resets the network's weights",
            "Exits the program",
            "Shows the help"
        ]
    }

    # Ein Pandas DataFrame zur Darstellung der Tabelle
    df = pd.DataFrame(help_data)
    print("--- Available Commands ---" + "-" * (61 - 26))
    # index=False entfernt die Zeilennummern (0, 1, 2...)
    print(df.to_string(index=False, justify='left',
                       formatters={'Command': lambda x: f"{x:<10}",
                                   'Description': lambda x: f"{x:<50}"}))
    # Trennlinie genau passend zur Tabellenbreite (10 + 1 + 50 = 61)
    print("-" * 61 + "\n")


# 1. INIT
# Erzeuge neuronales Netzwerkobjekt n mit der Anzahl der Pixel als Anzahl der Nodes im Input Layer.
# Wir setzen voraus, dass alle Bilder die gleiche Größe haben
# und eine Zahl von 0 bis 9 gefunden werden soll
my_nn = initiate_neuralnetwork(N_PIXEL, N_OUTPUT)

# 2. MAIN LOOP
while True:
    try:
        user_input = input("What do you want to do? "
                           "[Train|Test|Ask|Save|Load|Delete|Reset|End|Help]\n")
        match repm.REqual(user_input):
            case r'^[T|t]rain':
                # Benutzerabfrage
                try:
                    n_epochs = int(input("How many epochs do you want to train?\n"))
                    visualize = input("Do you want visualization? [True|False]\n")
                except ValueError as e:
                    print("Error in user input. Try again!")
                    break

                # Da beim Visualisieren ein anderer Ansatz gewählt wird, hier unterscheiden
                if visualize == "True":
                    accuracy_data = {}
                    test_list = read_dataset(TEST_DATASET)
                    accuracy = test_neuralnetwork(my_nn, test_list, False)
                    accuracy_data[0] = accuracy
                    print(f"Epoch {0}: Accuracy = {accuracy:.2%}")
                    # Trainieren mit jeweils einer Epoche für accuracy Daten
                    for epoch in range(n_epochs):
                        train_neuralnetwork(p_neuralnetwork=my_nn, epochs=1, print_output=False)
                        accuracy = test_neuralnetwork(my_nn, test_list, False)
                        accuracy_data[epoch + 1] = accuracy

                        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.2%}")

                    # Visualisieren
                    create_plot(accuracy_data)

                else:
                    # Neuronales Netzwerk trainieren
                    train_neuralnetwork(p_neuralnetwork=my_nn, epochs=n_epochs)
            case r'^[T|t]est':
                # Neuronales Netzwerk testen

                # Testdatensatz einlesen
                # (Besteht aus einer CSV mit einer Zahl und einem Bild für die Zahl pro Zeile)
                test_list = read_dataset(TEST_DATASET)
                test_neuralnetwork(my_nn, test_list)
            case r'^[A|a]sk':
                img_files = list_files(PIC_DIR, ".png")
                if img_files:
                    print(f"Found {len(img_files)} image(s):")
                    for img in img_files:
                        print(f"  -> {img}")
                filename = input("Please type in the file name to load: ").strip()
                if not filename:
                    print("That is not a valid filename. Try again!")
                    break
                if not filename.endswith(".png"):
                    filename = filename + ".png"
                img = il.ImageLoader(os.path.join(PIC_DIR, filename))
                img_values = img.get_pixel_values()
                guess = ask_neuralnetwork(my_nn, img_values, 0.4, True, True)
                if not guess == -1:
                    print(f"The neural network thinks this is a: {guess}.\n")
                else:
                    print("The neural network wasn't confident enough to make a guess.")
            case r'datasets':
                # For Debugging
                print("Test:")
                print_dataset(read_dataset(TEST_DATASET))
                print("Train:")
                print_dataset(read_dataset(TRAIN_DATASET))
            case r'^[S|s]ave':
                models = list_files(MODELS_DIR, ".npz")
                if models:
                    print(f"Found {len(models)} model(s):")
                    for model in models:
                        print(f"  -> {model}")
                filename = input("Please type in the file name "
                                 "(existing files will be overwritten!): ").strip()
                if not filename:
                    filename = "weights.npz"
                my_nn.save_weights(filename)
            case r'^[L|l]oad':
                # Modelle auflisten und prüfen, ob welche im Ordner sind
                models = list_files(MODELS_DIR, ".npz")
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
                    my_nn.load_weights(filename)
                else:
                    print("No models found.")
            case r'^[D|d]elete':
                # Find models in directory
                models = list_files(MODELS_DIR, ".npz")
                if models:
                    print(f"Found {len(models)} model(s):")
                    for model in models:
                        print(f"  -> {model}")
                    filename = input("Which model do you want to delete? [file name]: ").strip()
                    # Automatisch .npz anhängen, falls der Benutzer es vergessen hat
                    if not filename.endswith(".npz"):
                        filename += ".npz"

                    try:
                        # Dies löscht die Datei
                        os.remove(os.path.join(MODELS_DIR, filename))
                        print(f"Deleted {filename} from {MODELS_DIR}")
                    except FileNotFoundError as e:
                        print(f"File not found: {e.filename}")
                else:
                    print("No models found.")
            case r'^[R|r]eset':
                # This will reset the neural network by creating a new neural_network object
                print(f"Resetting CNN with ID: {hash(my_nn)}.")
                my_nn = initiate_neuralnetwork(N_PIXEL, N_OUTPUT)
                print(f"New CNN ID: {hash(my_nn)}.\n")
            case r'^[E|e]nd|^[E|e]xit':
                print("Exiting...")
                sys.exit(0)
            case r'^[H|h]elp':
                print_help()
            case _:
                print(f"Command not recognized: {user_input}. Type 'Help' for help.")
    except KeyboardInterrupt:
        print("User interrupted")
        sys.exit(0)
