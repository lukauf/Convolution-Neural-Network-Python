from cnn_model import CNNModel

def explain_layer(layer):
    print(f"\nCamada: {layer.name}")
    print(f"- Tipo: {layer.__class__.__name__}")

    try:
        shape = layer.output.shape
        print(f"- Saída: {shape}")

        if len(shape) == 4:
            print("  > Explicação do shape:")
            print(f"    - {shape[0]} → tamanho do batch")
            print("       (None significa que o valor será definido depois, quando o modelo for executado com fit(), evaluate() ou predict())")
            print(f"    - {shape[1]} → altura do mapa de ativação (pixels)")
            print(f"    - {shape[2]} → largura do mapa de ativação (pixels)")
            print(f"    - {shape[3]} → profundidade (número de filtros ou canais)")
        elif len(shape) == 2:
            print("  > Explicação do shape:")
            print(f"    - {shape[0]} → tamanho do batch")
            print("       (None será substituído por um número real no momento do treino ou teste)")
            print(f"    - {shape[1]} → número de neurônios da camada densa (dimensão da saída)")
    except Exception as e:
        print(f"- Saída: não disponível ({e})")

    print(f"- Nº de parâmetros treináveis: {layer.count_params()}")

    if hasattr(layer, 'activation'):
        print(f"- Função de ativação: {layer.activation.__name__}")

    if hasattr(layer, 'kernel_size'):
        print(f"- Tamanho do filtro: {layer.kernel_size}")

    if hasattr(layer, 'strides'):
        print(f"- Stride: {layer.strides}")

    if hasattr(layer, 'pool_size'):
        print(f"- Pooling size: {layer.pool_size}")

def print_summary_legend():
    print("\nLEGENDA DO SUMMARY:")
    print("------------------------------------------------------------")
    print("• Layer (type): Nome e tipo da camada (ex: Conv2D, Dense)")
    print("• Output Shape: Forma do tensor de saída da camada")
    print("   - 'None' representa o tamanho do batch (definido em tempo de execução)")
    print("   - Esse valor é determinado quando o modelo for chamado com fit(), predict() ou evaluate()")
    print("   - Nas camadas convolucionais: (None, altura, largura, canais)")
    print("   - Nas densas: (None, número de neurônios)")
    print("• Param #: Número total de parâmetros treináveis da camada")
    print("   - Exemplo: Conv2D com (3x3x1) filtros + 1 bias → (3×3×1 + 1)×num_filtros")
    print("------------------------------------------------------------")
    print("• Total params: soma de todos os parâmetros da rede")
    print("• Trainable params: parâmetros que serão atualizados no treino")
    print("• Non-trainable params: constantes (ex: camadas congeladas ou batchnorm em eval)")
    print("------------------------------------------------------------\n")

def main():
    cnn = CNNModel()
    model = cnn.model

    print("RESUMO DO MODELO (model.summary()):\n")
    model.summary()



    print("EXPLICAÇÃO DETALHADA DAS CAMADAS:")
    for layer in model.layers:
        explain_layer(layer)

    total_params = model.count_params()
    print(f"\nTotal de parâmetros treináveis: {total_params}")

if __name__ == "__main__":
    main()
