from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import numpy as np
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="multi or binary", required=True, choices=["multi", "binary"], default="multi")
parser.add_argument("-cl", "--convolutionLayers", help="Number of convolution layers", required=False, type=int, default=2)
parser.add_argument("-e", "--epochs", help="Number of epochs for training", type=int, default=5, required=False)
parser.add_argument("-dn", "--denseNeurons", help="Size", type=int, default=128, required=False)
parser.add_argument("-lr", "--learningRate", help="Size", type=float, default=0.001, required=False)

args = parser.parse_args()

# Carrega dados do Fashion MNIST da pasta Dataset/
(x_train, y_train), _ = load_fashion_mnist_ubyte("Dataset")

# Seleciona apenas duas classes para classificação binária
if args.mode == "binary":
    # Define os grupos de classes
    classes_1 = [0, 1, 2, 8, 4, 6, 3] # Classes que são calçados
    classes_0 = [5, 7, 9] # Classes que não são calçados
    classes_usadas = classes_1 + classes_0

    # Filtra apenas as classes usadas
    mask = np.isin(y_train, classes_usadas)
    x_train, y_train = x_train[mask], y_train[mask]

    # Reatribui rótulos: 1 para classes_1, 0 para classes_0
    y_train = np.isin(y_train, classes_1).astype("float32")

# Normaliza e adiciona canal
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]

# Prepara dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

# Instancia e treina o modelo
cnn = CNNModel(mode=args.mode, convolutionLayers=args.convolutionLayers, denseNeurons=args.denseNeurons)
cnn.compile(args.learningRate)
cnn.fit(train_ds, args.epochs)

# Salva os pesos
model_weights_filepath = f"./trained-models/cnn_fashion.{args.mode}.weights.h5"
cnn.save_weights(model_weights_filepath)
params_filepath = f"./trained-models/cnn_fashion.{args.mode}.params.txt"
cnn.save_params(params_filepath)