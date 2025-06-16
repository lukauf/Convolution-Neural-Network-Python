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

https://files09.oaiusercontent.com/file-AD93mLoGAytb89fHnmR4Uq?se=2025-06-16T02%3A43%3A05Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D299%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3D2f7c9b98-e9e6-4ad6-ad46-6d3545da5312.png&sig=wXIJCC2ia1WKEGPu/Kj2mU3uTNGt72xg1A1Ykw5a/CU%3D![image](https://github.com/user-attachments/assets/0ecccee5-36d1-48ee-8bcb-b7e663195ee1)




(O arquivo .h5 gerado pelo treino, é referente aos pesos ajustados pelo treino, você não precisa acessar esse arquivo, ele somente precisa ser gerado para que run_cnn.py seja executado com sucesso)
