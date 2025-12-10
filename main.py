import numpy


# Neuronales Netz Klassen Defintion
class neuralNetwork:

    # Initialisierung des neuronalen Netzes
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Anzahl der Knoten in der Eingabeschicht, versteckten Schicht und Ausgabeschicht
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Lernrate
        self.lr = learningrate
        pass

    # Training des neuronalen Netzes
    def train():
        pass

    # Abfrage des neuronalen Netzes
    def query():
        pass


# 1. Versuch ein kleines neuronales Netzobjekt zu erstellen mit 3 Knoten in jeder Schicht und einer Lernrate von 0,3
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print(numpy.random.rand(3, 3) - 0.5)

# Initialisierung der Gewichtungsmatrizen des neuronalen Netzes (wih und who) mit kleinen zufälligen Werten zwischen -0.5 und 0.5
# Zufällig, um die Symmetrie der Eingaben zu verhindern
# Ausgangspunkt für die Backpropagation
self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
self.who = numpy.random.rand(self.hnodes, self.inodes) - 0.5
