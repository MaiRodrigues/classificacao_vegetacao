import os
from PIL import Image
import argparse

def process_directory(input_dir, output_dir):
    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre todas as imagens no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            image_path = os.path.join(input_dir, filename)

            # Carregar a imagem TIFF
            image = Image.open(image_path)

            # Obter dimensões da imagem original
            width, height = image.size

            # Definir o número de colunas e linhas para o grid
            num_columns = 20
            num_rows = 10

            # Calcular as dimensões de cada subimagem
            subimage_width = width // num_columns
            subimage_height = height // num_rows

            # Dividir a imagem em subimagens
            for i in range(num_columns):  # Colunas
                for j in range(num_rows):  # Linhas
                    left = i * subimage_width
                    upper = j * subimage_height
                    right = left + subimage_width
                    lower = upper + subimage_height

                    # Garantir que as subimagens não ultrapassem os limites da imagem original
                    right = min(right, width)
                    lower = min(lower, height)

                    # Cortar e salvar a subimagem
                    subimage = image.crop((left, upper, right, lower))
                    subimage.save(os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_subimage_{j + 1}_{i + 1}.png"))

    print(f"Imagens salvas com sucesso no diretório '{output_dir}'!")

if __name__ == "__main__":
    # Configurar argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Divide images into subimages and save them.")
    parser.add_argument("--input", required=True, help="Path to input images directory")
    parser.add_argument("--output", required=True, help="Path to output directory")

    args = parser.parse_args()

    # Processar o diretório
    process_directory(args.input, args.output)
