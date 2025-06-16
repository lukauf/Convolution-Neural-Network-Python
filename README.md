# Projeto Python - Convolutional Neural Network (CNN)

Este projeto contém um exemplo básico de uma CNN usando Keras/TensorFlow para classificar dígitos do dataset MNIST.

## Como executar

1. Instale as dependências:

```bash
pip install requirements.txt
```

2. Execute o script para os pesos adquiridos do treinamento serem salvos:

```bash
python train_cnn.py
```

3. Execute o script para rodar a CNN com os pesos adquiridos no passo 2:

```bash
python run_cnn.py

```

(Quando executar a rede, os numeros que aparecem ao lado da barra de carregamento não é referente ao número de imagens processadas, mas sim ao número de batchs processados)



(O arquivo .h5 gerado pelo treino, é referente aos pesos ajustados pelo treino, você não precisa acessar esse arquivo, ele somente precisa ser gerado para que run_cnn.py seja executado com sucesso)
