""" Stellt die Klasse NeuralNetwork zur Verfügung. """

import os
import numpy
import scipy

# Neuronales Netz Klassen Definition
class NeuralNetwork:
    """
    Diese Klasse soll ein neuronales Netzwerk repräsentieren.
    Sie stellt Instanzmethoden zum Trainieren, Fragen, Speichern und
    Laden einer NeuralNetwork-Instanz bereit.
    """

    # Initialisierung des neuronalen Netzes
    def __init__(self, input_nodes=784, hidden_nodes=600, output_nodes=10, learning_rate=0.3):
        """
        Initialisierung der Gewichtungsmatrizen (wih und who) des neuronalen Netzes mit
        kleinen zufälligen Werten zwischen -0.5 und 0.5 (zufällig), um die Symmetrie
        der Eingaben zu verhindern.

        :param input_nodes: Anzahl der Eingabewerte das NN-Objekt erwartet
        :param hidden_nodes: Anzahl der Nodes in der Hidden Layer
        :param output_nodes: Anzahl der Ausgabe-Nodes
        :param learning_rate: Lernrate des NN-Objektes
        """
        # Anzahl der Knoten in der Eingabeschicht, der versteckten Schicht und Ausgabeschicht
        self.in_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.out_nodes = output_nodes
        # Ausgangspunkt für die Backpropagation: Die Funktion entnimmt ...
        # Stichproben aus einer Normalverteilung
        # Parameter: Mittelwert der Verteilung (0.0) Standardabweichung, Größe eines numpy-Arrays
        self.wih = numpy.random.normal(
            0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.in_nodes)
        )
        self.who = numpy.random.normal(
            0.0, pow(self.out_nodes, -0.5), (self.out_nodes, self.h_nodes)
        )
        # Lernrate
        self.lr = learning_rate
        # Lambda-Ausdruck, um die Aktivierungsfunktion zu speichern
        self.activation_function = lambda x: scipy.special.expit(x)

    # Training des neuronalen Netzes
    def train(self, inputs_list, targets_list):
        """
        Trainiert das neuronale Netzwerk.

        :param inputs_list: Eingabewerte mit denen trainiert wird
        :param targets_list: eigentlich erwartete Output-Werte des Netzwerks
        :return: None
        """
        # inputs in ein 2D überführen
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # Signale in die versteckte Schicht berechnen
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Signale aus der versteckten Schicht heraus berechnen
        hidden_outputs = self.activation_function(hidden_inputs)
        # Signale in den Output-Layer hinein berechnen
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Signale berechnen, die den Output-Layer verlassen
        final_outputs = self.activation_function(final_inputs)

        # Den Fehler zwischen Soll-Ergebnis und Ist-Ergebnis berechnen
        output_errors = targets - final_outputs
        # Backpropagierung: Fehler werden entsprechend der Verbindungsgewichte aufgeteilt
        # und für jeden Knoten der versteckten Schicht entsprechend zusammengefasst
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Aktualisierung der Gewichte zwischen den Verbindungen der ...
        # versteckten Schicht und der Ausgabeschicht
        # Genauer: Lernrate self.lr wird mit dem Rest multipliziert; ...
        # numpy.dot() für Matrizenmultiplikation
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs),
        )
        # Aktualisierung der Gewichte zwischen den Verbindungen der ...
        # Eingabeschicht und der versteckten Schicht
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs),
        )

    def query(self, inputs_list):
        """
        Abfrage des neuronalen Netzes (übernimmt die Eingabe in
        das neuronale Netz und liefert die Ausgabe des Netzes zurück)

        :param inputs_list:
        :return: Liefert einen Tupel der Ausgabe zurück
        (Anzahl der Elemente hängt von der Anzahl der Output Layers ab)
        """
        # Konvertierung der Eingaben in einen 2D-Array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # Berechnung der Signale in die versteckten Schichten hinein
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Berechnung der Signale aus der versteckten Schicht hinaus
        hidden_outputs = self.activation_function(hidden_inputs)
        # Berechnung der Signale in die Ausgabeschicht hinein
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Berechnung der Signale aus der Ausgabeschicht hinaus
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def save_weights(self, filename="weights.npz", path="models"):
        """
        Speichert die aktuellen Gewichtsmatrizen komprimiert in eine .npz Datei.

        :param filename: Name der Datei
        :param path: Pfad der Datei
        :return: None
        """
        try:
            # speichert die Weights komprimiert ab
            numpy.savez_compressed(os.path.join(path, filename), wih=self.wih, who=self.who)
            print(f"Saved as {filename}")
        except IOError as e:
            print(f"Error {e.strerror} while trying to save '{e.filename}'")

    def load_weights(self, filename="weights.npz", path="models"):
        """
        Lädt die Gewichtsmatrizen aus einer .npz Datei in das NN-Objekt.

        :param filename: Name der Datei
        :param path: Pfad der Datei
        :return: None
        """
        try:
            # öffnen des Archivs mittels vorgesehener Funktion aus numpy
            with numpy.load(os.path.join(path, filename)) as data:
                # Weights in die entsprechenden Variablen speichern
                self.wih = data['wih']
                self.who = data['who']
            print(f"File {filename} was loaded")
        except FileNotFoundError as e:
            print(f"Error {e.strerror}: File '{e.filename}' not found")
        except IOError as e:
            print(f"Unexpected error {e.strerror} while loading '{e.filename}'")
