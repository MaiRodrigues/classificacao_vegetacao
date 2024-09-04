import argparse
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser(description='Neural network model training with VGG16.')
    parser.add_argument('--csv', type=str, required=True, help='Path to the CSV file containing the data.')
    parser.add_argument('--model_output', type=str, required=True, help='Path to save the trained model, with .h5 or .keras extension.')
    return parser.parse_args()

# Leitura dos argumentos
args = parse_args()

# Verificar a extensão do arquivo e adicionar se necessário
if not (args.model_output.endswith('.h5') or args.model_output.endswith('.keras')):
    raise ValueError("A extensão do arquivo para salvar o modelo deve ser '.h5' ou '.keras'")

# Carregar o CSV
df = pd.read_csv(args.csv)

# Definindo os parâmetros das imagens
image_width = 160
image_height = 160
image_color_channel = 3  # Definindo canais de cor para preto e branco
image_color_channel_size = 255  # Tamanho máximo que cada canal de cor pode ter
image_size = (image_width, image_height)  # Tupla que armazena as dimensões da imagem
image_shape = image_size + (image_color_channel,)  # Tupla que combina as dimensões da imagem com o número de canais de cor

X = []
# Atualizar o carregamento da imagem para escala de cinza
for img in df['Imagem']:
    img = cv2.imread(str(img))
    img = cv2.resize(img, image_size)  # Usando image_size
    img = img / image_color_channel_size  # Usando image_color_channel_size
    X.append(img)

rotulos = ["Sim", "Não"]

y = df['Vegetacao']

# Divisão em treino, teste e validação
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.2, random_state=0)

# Carregar a arquitetura VGG16
base_model = VGG16(input_shape=image_shape, include_top=False, weights='imagenet')

for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=image_shape),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = Sequential()
model.add(data_augmentation)
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(rotulos), activation='sigmoid'))

# Mudança nas estruturas em arrays NumPy bidimensionais
X_train = np.stack(X_train, axis=0)
X_val = np.stack(X_val, axis=0)

# Codificação one-hot
y_train_one_hot = to_categorical(y_train, num_classes=2)
y_val_one_hot = to_categorical(y_val, num_classes=2)

# Definições do modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['acc']
)

# Definindo o EarlyStopping e ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Dados de treino, validações e quantidade de iterações
history = model.fit(X_train, y_train_one_hot, epochs=100, validation_data=(X_val, y_val_one_hot),
                    callbacks=[early_stopping, reduce_lr])

X_test = np.stack(X_test, axis=0)
y_test_one_hot = to_categorical(y_test, num_classes=2)

# Avaliação com os dados de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test_one_hot)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

model.save(args.model_output + '.keras')  # Salva o modelo no formato Keras
