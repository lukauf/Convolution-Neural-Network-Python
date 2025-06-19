from cnn_model import CNNModel

def main():
    # Cria instância da classe e acessa o modelo interno
    cnn = CNNModel()
    cnn.model.summary()  # Chama o summary diretamente no modelo do TensorFlow

if __name__ == "__main__":
    main()
