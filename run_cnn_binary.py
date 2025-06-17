from load_ubyte import load_fashion_mnist_ubyte
from cnn_model_binary import CNNModel_Binary
import tensorflow as tf
import numpy as np

# Carrega dados de teste
_, (x_test, y_test) = load_fashion_mnist_ubyte("Dataset")

# Filtra apenas as duas classes usadas no treino (ex: 0 e 1)
classes_usadas = [0, 1]
mask = np.isin(y_test, classes_usadas)
x_test, y_test = x_test[mask], y_test[mask]

# Reatribui rótulos para 0 e 1
y_test = (y_test == classes_usadas[1]).astype("float32")

# Normaliza e adiciona canal
x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., tf.newaxis]

# Prepara dataset de teste
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Instancia o modelo e carrega pesos treinados
cnn = CNNModel_Binary()
cnn.compile()
cnn.load_weights("cnn_binary.weights.h5")

# Avalia
loss, acc = cnn.evaluate(test_ds)

# Predição de uma imagem
pred = cnn.predict(x_test[:1])[0][0]
pred_class = 1 if pred > 0.5 else 0
print(f"Classe prevista: {pred_class}, Classe esperada: {int(y_test[0])}")

# Nomes das classes (ajustado para binário)
class_names = ['Shoe', 'Not Shoe']

# Faz predições para todo o x_test
predictions = cnn.predict(x_test)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Imprime previsões em linguagem humana
print("\nResultado das predições no conjunto de teste:")
for i in range(len(x_test)):
    pred_idx = predicted_classes[i]
    true_idx = int(y_test[i])
    print(f"Exemplo {i+1:5d} - Previsto: {class_names[pred_idx]:10s} ({pred_idx}) | Esperado: {class_names[true_idx]:10s} ({true_idx})")
    print(f"Acurácia: {acc * 100:.2f}%")
