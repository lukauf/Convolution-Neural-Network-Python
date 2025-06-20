# Projeto Python - Convolutional Neural Network (CNN)

Este projeto contém um exemplo básico de uma CNN usando Keras/TensorFlow para classificar dígitos do dataset MNIST.

## Como executar

1. Instale as dependências:

```bash
pip install -r requirements.txt
```

2. Execute o script para os pesos adquiridos do treinamento serem salvos:

```bash
python train_cnn.py -m <mode> -cl <convolution layers> -e <epochs> -dn <dense neurons> -lr <learning rate>
```
mode: multi ou binary, para classificações multiclasse ou binárias<br>
convolution layers: número de camadas de convolução e pooling
epochs: número de épocas de treinamento
dense neurons: número de neurons da camada densa
learning rate: taxa de treinamento

3. Execute o script para rodar a CNN com os pesos adquiridos no passo 2:

```bash
python run_cnn.py -m <mode>

```
mode: multi ou binary, para classificações multiclasse ou binárias<br>. É necessário treinar a rede do tipo certo antes de rodar.

4. Execute o script para ver um resumo sobre a estrutura

```bash
python Summary_multiclass.py -m <mode>

``` 
mode: multi ou binary, para classificações multiclasse ou binárias<br>. É necessário treinar a rede do tipo certo antes de rodar.

(Para visualizar 10 imagens do dataset, execute o script visualize.py)

```bash
python visualize.py

```

(Quando executar a rede, os numeros que aparecem ao lado da barra de carregamento não são referentes ao número de imagens processadas, mas sim ao número de batches processados)



(O arquivo .h5 gerado pelo treino, é referente aos pesos ajustados pelo treino, você não precisa acessar esse arquivo, ele somente precisa ser gerado para que run_cnn.py seja executado com sucesso)