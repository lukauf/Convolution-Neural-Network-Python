import numpy as np
import matplotlib.pyplot as plt
import struct

# Caminhos para os arquivos .ubyte
IMG_PATH = "Dataset/t10k-images-idx3-ubyte"
LBL_PATH = "Dataset/t10k-labels-idx1-ubyte"

# Nomes das classes
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Função para carregar imagens
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return data

# Função para carregar rótulos
def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Carrega dados
images = load_images(IMG_PATH)
labels = load_labels(LBL_PATH)

# Mostra 10 imagens
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
