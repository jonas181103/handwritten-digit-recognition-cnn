import matplotlib.pyplot
import numpy as np

data_file = open(
    "/Users/jonas/Arbeit/Code Projekte/Klassifikation von Handschrift-Bilddaten mit einem künstlichen neuronalen Netz/data/raw/mnist_data/Testdaten/mnist_train_100.csv",
    "r",
)
data_list = data_file.readlines()
data_file.close()

print(len(data_list))
print(data_list[0])

all_values = data_list[0].split(",")
image_array = np.asarray(all_values[1:], dtype="float").reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation="None")
matplotlib.pyplot.show()
