import matplotlib.pyplot as plt
import numpy
import src.NeuralNetwork as NeuralNetwork
import scipy.special  # Sigmoid-Funktion


def initiateNeuralNetwork(input_nodes:int, output_nodes:int, learningrate:float = 0.3):
    """Berechne die Anzahl der Nodes für das neuronale Netzwerk anhand von Input und Output."""
    input_nodes = input_nodes
    hidden_nodes = int((input_nodes+output_nodes)//(3/2))
    output_nodes = output_nodes
    learning_rate = learningrate
    neural_network = NeuralNetwork.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    return neural_network

# Testnetzwerk
data_file = open("data/raw/mnist_data/Testdaten/mnist_test_10.csv", "r")

data_list = data_file.readlines()
data_file.close()

print(len(data_list))
print(data_list[1])

all_values = data_list[1].split(",")
image_array = numpy.asarray(all_values[1:], dtype="float").reshape((28, 28))
plt.imshow(image_array, cmap="Greys", interpolation="None")
plt.show()

# Erzeuge neuronales Netzwerkobjekt n mit der Anzahl der Pixel als Anzahl der Nodes im Input Layer
pixel = len(all_values[1:])
n_output = len(data_list)
n = initiateNeuralNetwork(pixel, n_output)

output = n.query((numpy.asarray(all_values[1:], dtype="float") / 255.0 * 0.99) + 0.01)
print(output)
