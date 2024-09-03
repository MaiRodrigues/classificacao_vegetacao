import os
import cv2
import numpy as np
import pandas as pd
import argparse


# Definir a função para calcular o Índice de Excesso de Verde (ExG)
def calculate_excess_green(image):
    # Converter a imagem para float para maior precisão
    image = image.astype('float32')

    # Separar a imagem em seus componentes Vermelho, Verde e Azul
    R, G, B = cv2.split(image)

    # Calcular o Índice de Excesso de Verde (ExG)
    exg = 2 * G - R - B

    # Aplicar limiar para classificar vegetação
    vegetation_mask = exg > 0

    return vegetation_mask


def main(input_directory, output_directory):
    # Listar todos os arquivos de imagem no diretório de entrada
    image_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if
                   file.endswith(('jpg', 'png', 'jpeg'))]

    # Criar um DataFrame e processar cada imagem
    data = {'Imagem': [], 'Vegetacao': []}

    for image_file in image_files:
        # Ler a imagem
        image = cv2.imread(image_file)

        # Calcular a máscara de vegetação
        vegetation_mask = calculate_excess_green(image)

        # Converter a máscara booleana em uma imagem em escala de cinza de 8 bits (0 ou 255)
        vegetation_image = (vegetation_mask * 255).astype(np.uint8)

        # Salvar a imagem de vegetação no diretório de saída
        filename = os.path.basename(image_file)
        output_file = os.path.join(output_directory, filename)
        cv2.imwrite(output_file, vegetation_image)

        # Determinar se a maioria dos pixels são de vegetação
        if np.mean(vegetation_mask) > 0.25:
            vegetation_label = 1
        else:
            vegetation_label = 0

        # Adicionar ao DataFrame
        data['Imagem'].append(image_file)
        data['Vegetacao'].append(vegetation_label)

    # Converter o dicionário em um DataFrame
    df = pd.DataFrame(data)

    # salvar o DataFrame em um arquivo CSV
    df.to_csv(os.path.join(output_directory, 'vegetation_labels.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binarize images based on Excess Green Index and save the results.")
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing the images.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the directory where segmented images and CSV will be saved.')

    args = parser.parse_args()

    main(args.input, args.output)
