from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import numpy as np
import tensorflow as tf
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="multi or binary", required=True, choices=["multi", "binary"], default="multi")
parser.add_argument("-e", "--epochs", help="Number of epochs for training", type=int, default=5, required=False)
parser.add_argument("-dn", "--denseNeurons", help="Number of neurons in the dense layer of the CNN", type=int, default=128, required=False)
parser.add_argument("-lr", "--learningRate", help="Learning Rate for training", type=float, default=0.001, required=False)
parser.add_argument("-cm", "--confusionMatrix", help="Show confusion matrix after training", type=bool, default=False, required=False)

args = parser.parse_args()

# Carrega dados do Fashion MNIST da pasta Dataset/
(x_train, y_train), (x_test, y_test) = load_fashion_mnist_ubyte("Dataset")

# Seleciona apenas duas classes para classificação binária
if args.mode == "binary":
    classes_1 = [0, 1, 2, 8, 4, 6, 3]
    classes_0 = [5, 7, 9]
    classes_usadas = classes_1 + classes_0

    # Treino
    mask_train = np.isin(y_train, classes_usadas)
    x_train, y_train = x_train[mask_train], y_train[mask_train]
    y_train = np.isin(y_train, classes_1).astype("float32")

    # Teste
    mask_test = np.isin(y_test, classes_usadas)
    x_test, y_test = x_test[mask_test], y_test[mask_test]
    y_test = np.isin(y_test, classes_1).astype("float32")

    class_names = ['Not Shoe', 'Shoe']
else:
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

# Normaliza e adiciona canal
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Prepara datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(10000).batch(32)

# Instancia e treina o modelo
cnn = CNNModel(mode=args.mode, denseNeurons=args.denseNeurons)
cnn.compile(args.learningRate)
cnn.fit(train_ds, args.epochs)

# Salva os pesos
if not tf.io.gfile.exists("./trained-models"):
    tf.io.gfile.makedirs("./trained-models")
model_weights_filepath = f"./trained-models/cnn_fashion.{args.mode}.weights.h5"
cnn.save_weights(model_weights_filepath)
params_filepath = f"./trained-models/cnn_fashion.{args.mode}.params.txt"
cnn.save_params(params_filepath)

# Matriz de confusão
if args.confusionMatrix:
    predictions = cnn.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1).astype("int32").flatten()
    y_test = y_test.astype("int32")

    cm = confusion_matrix(y_test, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Matriz de Confusão - Modo: {args.mode.capitalize()}")
    plt.tight_layout()
    plt.show()
