from dataset_loader import FashionMNISTLoader
from cnn_model import CNNModel

# Caminhos atualizados para a pasta Dataset/
train_img = "Dataset/train-images-idx3-ubyte"
train_lbl = "Dataset/train-labels-idx1-ubyte"
test_img = "Dataset/t10k-images-idx3-ubyte"
test_lbl = "Dataset/t10k-labels-idx1-ubyte"

# Carregando os dados
train_loader = FashionMNISTLoader(train_img, train_lbl)
test_loader = FashionMNISTLoader(test_img, test_lbl)

# tf.data.Dataset
train_ds = train_loader.get_dataset(batch_size=64)
test_ds = test_loader.get_dataset(batch_size=64, shuffle=False)

# Criar, compilar e treinar a CNN
cnn = CNNModel()
cnn.compile()
cnn.fit(train_ds, val_ds=test_ds, epochs=5)

# Avaliar
loss, acc = cnn.evaluate(test_ds)
print(f"Acurácia de teste: {acc * 100:.2f}%")
cnn.save_weights("cnn_fashion_weights.h5")
