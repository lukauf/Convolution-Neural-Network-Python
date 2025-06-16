from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import tensorflow as tf

# Carrega dados do Fashion MNIST da pasta Dataset/
(x_train, y_train), _ = load_fashion_mnist_ubyte("Dataset")

# Normaliza e adiciona canal
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]

# Prepara dataset
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

# Instancia e treina o modelo
cnn = CNNModel()
cnn.compile()
cnn.fit(train_ds, epochs=5)

# Salva os pesos
cnn.save_weights()
