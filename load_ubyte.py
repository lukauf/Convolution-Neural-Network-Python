import os
import numpy as np
import struct

def load_images(path):
    with open(path, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

def load_labels(path):
    with open(path, 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

def load_fashion_mnist_ubyte(dataset_dir="Dataset"):
    train_images = load_images(os.path.join(dataset_dir, "train-images-idx3-ubyte"))
    train_labels = load_labels(os.path.join(dataset_dir, "train-labels-idx1-ubyte"))
    test_images = load_images(os.path.join(dataset_dir, "t10k-images-idx3-ubyte"))
    test_labels = load_labels(os.path.join(dataset_dir, "t10k-labels-idx1-ubyte"))

    return (train_images, train_labels), (test_images, test_labels)
