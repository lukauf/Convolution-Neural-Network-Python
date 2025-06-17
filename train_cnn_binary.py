from load_ubyte import load_fashion_mnist_ubyte
from cnn_model_binary import CNNModel_Binary
import tensorflow as tf
import numpy as np

# Carrega dados do Fashion MNIST da pasta Dataset/
(x_train, y_train), _ = load_fashion_mnist_ubyte("Dataset")

# Seleciona apenas duas classes (ex: classe 0 e classe 1)
classes_usadas = [0, 1]
mask = np.isin(y_train, classes_usadas)
x_train, y_train = x_train[mask], y_train[mask]

# Reatribui os rótulos para ficarem 0 ou 1
y_train = (y_train == classes_usadas[1]).astype("float32")

# Normaliza e adiciona canal
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]

# Prepara dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

# Instancia e treina o modelo
cnn = CNNModel_Binary()
cnn.compile()
cnn.fit(train_ds, epochs=5)

# Salva os pesos
cnn.save_weights("cnn_binary.weights.h5")
