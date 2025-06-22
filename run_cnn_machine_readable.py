from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import tensorflow as tf
import numpy as np

neuron_amount_list = [64, 128, 256, 512, 1024]
epoch_list = [2, 5, 8, 10, 15, 20, 25, 30]

# Carrega dados do Fashion MNIST da pasta Dataset/
(x_train, y_train), _ = load_fashion_mnist_ubyte("Dataset")

# Normaliza e adiciona canal
x_train = x_train.astype("float32") / 255.0
x_train = x_train[..., tf.newaxis]

# Carrega dados de teste
_, (x_test, y_test) = load_fashion_mnist_ubyte("Dataset")

# Normaliza e adiciona canal (tensor flow precisa do canal explicito)
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Prepara dataset de teste
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

for neuron_amount in neuron_amount_list:
    for epoch in epoch_list:
        # Prepara dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

        # Instancia e treina o modelo
        cnn = CNNModel(mode="multi", denseNeurons=neuron_amount)
        cnn.compile()
        cnn.fit(train_ds, epochs=epoch)

        # Avalia
        loss, acc = cnn.evaluate(test_ds)

        # Predição de uma imagem
        pred = cnn.predict(x_test[:1])
        #print(f"Classe prevista: {np.argmax(pred)}, Classe esperada: {y_test[0]}")

        # Mapeamento das classes
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        # Faz predições para todo o x_test
        predictions = cnn.predict(x_test)
        predicted_classes = np.argmax(predictions, axis=1)

        # Imprime previsões em linguagem humana
        print(f"neuron_amount: {neuron_amount}, epochs: {epoch}")
        print("\nexemplo, previsto_classe, esperado_classe")
        for i in range(len(x_test)):
            pred_idx = predicted_classes[i]
            true_idx = y_test[i]
            print(f"{i+1}, {class_names[pred_idx]}, {class_names[true_idx]}")
            #print(f"Acurácia: {acc * 100:.2f}%")

