import numpy
import scipy.special  # Sigmoid-Funktion


# Neuronales Netz Klassen Definition
class NeuralNetwork:

    # Initialisierung des neuronalen Netzes
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Anzahl der Knoten in der Eingabeschicht, der versteckten Schicht und Ausgabeschicht
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # Initialisierung der Gewichtungsmatrizen des neuronalen Netzes (wih und who) mit kleinen zufälligen Werten zwischen -0.5 und 0.5
        # Zufällig um die Symmetrie der Eingaben zu verhindern
        # Ausgangspunkt für die Backpropagation
        # die Funktion entnimmt Stichproben aus einer Normalverteilung
        # Parameter: Mittelwert der Verteilung (0.0), Standardabweichung, Größe eines numpy-Arrays
        self.wih = numpy.random.normal(
            0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)
        )
        self.who = numpy.random.normal(
            0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)
        )
        # Lernrate
        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # Training des neuronalen Netzes
    def train(self, inputs_list, targets_list):
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

        # Aktualisierung der Gewichte zwischen den Verbindungen der versteckten Schicht und der Ausgabeschicht
        # Genauer: Lernrate self.lr wird mit dem Rest multipliziert; numpy.dot() für Matrizenmultiplikation
        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs),
        )
        # Aktualisierung der Gewichte zwischen den Verbindungen der Eingabeschicht und der versteckten Schicht
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs),
        )
        pass

    # Abfrage des neuronalen Netzes (übernimmt die Eingabe in das neuronale Netz und liefert die Ausgabe des Netzes zurück)
    def query(self, inputs_list):
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


# 1. Versuch ein kleines neuronales Netzobjekt zu erstellen mit 3 Knoten in jeder Schicht und einer Lernrate von 0,3
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(n.query([1.0, 0.5, -1.5]))
