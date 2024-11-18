import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
import os

# Defina o caminho para as pastas de treinamento e validação
train_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\train'
validation_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\validation'

# Carregamento das imagens com redimensionamento e rótulos de pixel para segmentação
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode=None
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode=None
)

# Normalizar as imagens de entrada e preparar as máscaras binárias
train_dataset = train_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x)))  # Transformar rótulos em escala de cinza
validation_dataset = validation_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x)))

# Modelo CNN para segmentação
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),

    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Saída binária
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Salvando o modelo
model.save('prick_test_segmentation_model.h5')

def detect_bumps_opencv(img_path):
    # Carrega a imagem
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (600, 400))  # Redimensiona para facilitar o processamento

    # Converte para escala de cinza
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Suaviza a imagem para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Detecta bordas usando o Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Fecha as bordas para formar contornos completos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Encontra os contornos na imagem
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar tamanhos dos calombos
    lump_sizes = []

    # Filtra contornos e desenha na imagem original
    min_area = 50  # Área mínima para contornos
    max_area = 1000  # Área máxima para contornos
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Adiciona o tamanho do calombo à lista
            lump_sizes.append(area)

            # Calcula o retângulo delimitador do contorno
            x, y, w, h = cv2.boundingRect(contour)

            # Desenha o retângulo ao redor do calombo
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Anota o tamanho do calombo
            cv2.putText(original_img, f"{area:.1f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Salva e exibe a imagem anotada
    output_path = 'detected_bumps.jpg'
    cv2.imwrite(output_path, original_img)

    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Calombos Detectados")
    plt.show()

    return lump_sizes


# Teste com uma nova imagem
test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado1.png'
lump_sizes = detect_bumps_opencv(test_image_path)
print("Tamanhos dos calombos detectados:", lump_sizes)
