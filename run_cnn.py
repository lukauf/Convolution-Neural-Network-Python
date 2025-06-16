from load_ubyte import load_fashion_mnist_ubyte
from cnn_model import CNNModel
import tensorflow as tf
import numpy as np

# Carrega dados de teste
_, (x_test, y_test) = load_fashion_mnist_ubyte("Dataset")

# Normaliza e adiciona canal (tensor flow precisa do canal explicito)
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Prepara dataset de teste
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Instancia o modelo e carrega pesos treinados
cnn = CNNModel()
cnn.compile()
cnn.load_weights()

# Avalia
loss, acc = cnn.evaluate(test_ds)

# Predição de uma imagem
pred = cnn.predict(x_test[:1])
print(f"Classe prevista: {np.argmax(pred)}, Classe esperada: {y_test[0]}")

# Mapeamento das classes
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Faz predições para todo o x_test
predictions = cnn.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)

# Imprime previsões em linguagem humana
print("\nResultado das predições no conjunto de teste:")
for i in range(len(x_test)):
    pred_idx = predicted_classes[i]
    true_idx = y_test[i]
    print(f"Exemplo {i+1:5d} - Previsto: {class_names[pred_idx]:12s} ({pred_idx}) | Esperado: {class_names[true_idx]:12s} ({true_idx})")
    print(f"Acurácia: {acc * 100:.2f}%")

