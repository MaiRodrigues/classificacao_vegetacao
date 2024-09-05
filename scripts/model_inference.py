import argparse
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Classificar imagens de um diretório usando um modelo treinado.')
    parser.add_argument('--image_dir', type=str, required=True, help='Caminho para o diretório contendo as imagens.')
    parser.add_argument('--modelpath', type=str, required=True, help='Caminho para o arquivo do modelo treinado (.h5 ou .keras).')
    parser.add_argument('--output_dir', type=str, required=True, help='Caminho para salvar os resultados da classificação.')
    return parser.parse_args()

def load_image(image_path, target_size=(160, 160)):
    image = Image.open(image_path)

    # Converter para RGB se a imagem tiver um canal alfa (RGBA)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Redimensionar e normalizar a imagem
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalizar para o intervalo [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Adicionar dimensão de lote (batch)
    return image_array

def classify_images(model, image_dir, output_dir):
    class_labels = [0, 1]  # Modificar com base nos rótulos de classe do seu modelo
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image_array = load_image(image_path)

            predictions = model.predict(image_array)
            predicted_class = class_labels[np.argmax(predictions)]

            # Salvar resultado no diretório de saída
            result_file = os.path.join(output_dir, f"{filename}_result.txt")
            with open(result_file, 'w') as f:
                f.write(f"Imagem: {filename}, Classe Prevista: {predicted_class}\n")
            print(f"Classificou {filename}: {predicted_class}")

if __name__ == '__main__':
    args = parse_args()

    # Carregar o modelo treinado
    model = load_model(args.modelpath)

    # Classificar as imagens do diretório
    classify_images(model, args.image_dir, args.output_dir)
