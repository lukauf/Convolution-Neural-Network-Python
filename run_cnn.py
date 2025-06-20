from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import argparse
import os
import fnmatch
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--mode", help="multi or binary", required=True, choices=["multi", "binary"], default="multi")

args = parser.parse_args()

# Carrega dados de teste
_, (x_test, y_test) = load_fashion_mnist_ubyte("Dataset")

# Filtra apenas as duas classes para classificação binária
class_names = None
if args.mode == "binary":
    # Define os grupos de classes
    classes_1 = [0, 1, 2, 8, 4, 6, 3]
    classes_0 = [5, 7, 9]
    classes_usadas = classes_1 + classes_0

    # Filtra apenas as classes usadas
    mask = np.isin(y_test, classes_usadas)
    x_test, y_test = x_test[mask], y_test[mask]

    # Reatribui rótulos: 1 para classes_1, 0 para classes_0
    y_test = np.isin(y_test, classes_1).astype("float32")

    class_names = ['Shoe', 'Not Shoe']
else:
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

# Normaliza e adiciona canal (tensor flow precisa do canal explicito)
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Prepara dataset de teste
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Instancia o modelo e carrega pesos treinados
params_filepath = f"./trained-models/cnn_fashion.{args.mode}.params.txt"
cnn = CNNModel(params_filepath=params_filepath)
cnn.compile(0.001)  # learning rate padrão
weightFileName = f'./trained-models/cnn_fashion.{args.mode}.weights.h5'
cnn.load_weights(weightFileName)

# Avalia
loss, acc = cnn.evaluate(test_ds)

# Predição de uma imagem
pred = cnn.predict(x_test[:1])
print(f"Classe prevista: {np.argmax(pred)}, Classe esperada: {y_test[0]}")



# Faz predições para todo o x_test
predictions = cnn.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1).astype("int32").flatten()

# Imprime previsões em linguagem humana
print("\nResultado das predições no conjunto de teste:")
for i in range(len(x_test)):
    pred_idx = predicted_classes[i]
    true_idx = int(y_test[i])
    print(f"Exemplo {i+1:5d} - Previsto: {class_names[pred_idx]:12s} ({pred_idx}) | Esperado: {class_names[true_idx]:12s} ({true_idx})")
    print(f"Acurácia: {acc * 100:.2f}%")
