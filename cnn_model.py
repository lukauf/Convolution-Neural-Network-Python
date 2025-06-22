import tensorflow as tf
from keras import layers, models

class CNNModel:
    mode = None
    denseNeurons = None
    model = None

    def __init__(self=None, mode=None, denseNeurons=None, params_filepath=None):
        if params_filepath:
            self.load_params(params_filepath)
        else:
            if mode is None or denseNeurons is None:
                raise ValueError("mode and denseNeurons must be provided if params_filepath is not given.")
            self.mode = mode
            self.denseNeurons = denseNeurons

        self.model = self._build_model()

    def _build_model(self):
        if self.mode == "binary":
            # Se for binaria, sigmoid para 2 classes
            finalLayer = layers.Dense(2, activation='sigmoid')
        else:
            # Se for multi-classe, usamos softmax para 10 classes
            finalLayer = layers.Dense(10, activation='softmax')
        model = models.Sequential([
            # Entrada da imagem: 28x28x1 (tons de cinza)
            layers.Input(shape=(28, 28, 1)),

            # Camada convolucional 1
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),  # reduz 28x28 → 14x14

            # Camada convolucional 2
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D((2, 2)),  # reduz 14x14 → 7x7

            # Camadas densas (MLP)
            layers.Flatten(),  # transforma 7x7x64 em vetor de 3136 elementos, entao a mlp vai ter 3136 neuronios na camada de entrada
            layers.Dense(self.denseNeurons, activation='relu'), #128 neuronios (numero arbitrario) na camada escondida
            finalLayer
        ])
        return model

    def compile(self, learning_rate):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #adam é um otimizador responsável por atualizar os pesos da rede, com base nos gradientes calculados pelo backpropagation
            loss='sparse_categorical_crossentropy', #loss é a funcao de perda
                                                    #sparse_categorical_crossentropy é o calculo de erro especifico para rotulos inteiros e nao one hot encoded
            metrics=['accuracy']#somente para monitoramento, nesse caso é a acuracia
        )

    def fit(self, train_ds, epochs=5):  # esse metodo treina a rede neural, com condições de parada de 99% de acurácia
        class StopAtAccuracy(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                acc = logs.get("accuracy")
                if acc is not None and acc >= 0.99:
                    print(f"\nParando treino: acurácia atingiu {acc*100:.2f}%")
                    self.model.stop_training = True

        self.model.fit(
            train_ds,
            epochs=epochs,
            callbacks=[StopAtAccuracy()]
        )

    def evaluate(self, test_ds):  #testa o desempenho final da rede em dados nunca vistos (test_ds)
        #calcula e retorna o loss e accuracy
        #nao altera pesos
        return self.model.evaluate(test_ds)

    def predict(self, input_batch): #usa a rede treinada para prever a classe de novas imagens
        return self.model.predict(input_batch)

    def save_weights(self, path): #pesos finais sao salvos para nao termos que refazer o treino toda vez que quisermos usar a rede
        self.model.save_weights(path)

    def load_weights(self, path): #carrega peso salvo
        self.model.load_weights(path)

    def save_params(self, params_filepath):
        with open(params_filepath, "w") as f:
            f.write(f"mode={self.mode}\n")
            f.write(f"denseNeurons={self.denseNeurons}\n")
            
    def load_params(self, params_filepath):
        params = {}
        with open(params_filepath, "r") as f:
            for line in f:
                key, value = line.strip().split("=")
                params[key] = value

        self.mode = params.get("mode", "multi")
        self.epochs = int(params.get("epochs", 5))
        self.denseNeurons = int(params.get("denseNeurons", 128))
        self.learningRate = float(params.get("learningRate", 0.001))