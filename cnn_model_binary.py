import tensorflow as tf
from tensorflow.keras import layers, models

class CNNModel_Binary:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            # Entrada da imagem: 28x28x1 (tons de cinza)
            layers.Input(shape=(28, 28, 1)),

            # Camada convolucional 1
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),  # reduz 28x28 → 14x14

            # Camada convolucional 2
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),  # reduz 14x14 → 7x7

            # Camadas densas (MLP)
            layers.Flatten(),  # transforma 7x7x64 em vetor de 3136 elementos
            layers.Dense(128, activation='relu'), #128 neuronios (numero arbitrario) na camada escondida
            layers.Dense(2, activation='sigmoid')  # 2 classes do dataset Fashion MNIST
        ])
        return model

    def compile(self, learning_rate=0.001):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #adam é um otimizador responsável por atualizar os pesos da rede, com base nos gradientes calculados pelo backpropagation
            loss='sparse_categorical_crossentropy', #loss é a funcao de perda
                                                    #sparse_categorical_crossentropy é o calculo de erro especifico para rotulos inteiros e nao one hot encoded
            metrics=['accuracy']#somente para monitoramento, nesse caso é a acuracia
        )

    def fit(self, train_ds, epochs=5): #esse metodo treina a rede neural
        #train ds: conjunto de dados de treinamento(imagens + rotulos)
        #epochs: numero de epocas
        self.model.fit(train_ds, epochs=epochs) #chamada da biblioteca

    def evaluate(self, test_ds):  #testa o desempenho final da rede em dados nunca vistos (test_ds)
        #calcula e retorna o loss e accuracy
        #nao altera pesos
        return self.model.evaluate(test_ds)

    def predict(self, input_batch): #usa a rede treinada para prever a classe de novas imagens
        return self.model.predict(input_batch)

    def save_weights(self, path='cnn_fashion.binary.weights.h5'): #pesos finais sao salvos para nao termos que refazer o treino toda vez que quisermos usar a rede
        self.model.save_weights(path)

    def load_weights(self, path='cnn_fashion.binary.weights.h5'): #carrega peso salvo
        self.model.load_weights(path)
