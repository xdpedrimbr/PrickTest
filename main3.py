import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
import os

# # Defina o caminho para as pastas de treinamento e validação
# train_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\train'
# validation_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\validation'

# # Carregamento das imagens com redimensionamento e rótulos de pixel para segmentação
# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     train_dir,
#     image_size=(150, 150),
#     batch_size=32,
#     label_mode=None
# )

# validation_dataset = tf.keras.utils.image_dataset_from_directory(
#     validation_dir,
#     image_size=(150, 150),
#     batch_size=32,
#     label_mode=None
# )

# # Normalizar as imagens de entrada e preparar as máscaras binárias
# train_dataset = train_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x)))  # Transformar rótulos em escala de cinza
# validation_dataset = validation_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x)))

# # Modelo CNN para segmentação
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
#     layers.MaxPooling2D((2, 2), padding='same'),

#     layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#     layers.UpSampling2D((2, 2)),

#     layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Saída binária
# ])

# # Compilação do modelo
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Treinamento do modelo
# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=10
# )

# # Salvando o modelo
# model.save('prick_test_segmentation_model.h5')

def detect_and_save_protuberances(img_path, output_dir):
    # Cria o diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Carrega a imagem original
    original_img = cv2.imread(img_path)
    original_img_resized = cv2.resize(original_img, (600, 400))  # Ajuste do tamanho da imagem

    # Converte para escala de cinza
    gray = cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2GRAY)

    # Suaviza a imagem para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Aplica o operador Laplace para detectar elevações
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # Converte para valores absolutos

    # Normaliza os valores para melhorar o contraste
    normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    # Aplica limiar para criar uma máscara binária
    _, mask = cv2.threshold(normalized, 30, 255, cv2.THRESH_BINARY)

    # Refinamento da máscara
    kernel = np.ones((5, 5), np.uint8)
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fecha pequenos buracos
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)  # Remove ruídos pequenos

    # Encontra contornos na máscara refinada
    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar tamanhos dos calombos
    lump_sizes = []

    # Contador para imagens das protuberâncias
    image_count = 0

    # Fatores para aumentar a área do círculo (ajuste o valor conforme necessário)
    scale_factor = 1.5  # Fator para aumentar o tamanho do círculo

    # Filtra contornos com base em tamanho e desenha círculos na imagem original
    min_area = 50  # Área mínima para protuberâncias
    max_area = 190 # Área máxima para protuberâncias
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Adiciona o tamanho da protuberância à lista
            lump_sizes.append(area)

            # Calcula o centro e o raio do círculo mínimo
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx), int(cy))
            radius = int(radius)

            # Aumenta o tamanho do círculo aplicando um fator de escala
            scaled_radius = int(radius * scale_factor)  # Expande o raio

            # Desenha o círculo aumentado ao redor da protuberância
            cv2.circle(original_img_resized, center, scaled_radius, (0, 255, 0), 2)

            # Recorta a região da protuberância (ROI)
            x, y, w, h = cv2.boundingRect(contour)
            roi = original_img_resized[y:y + h, x:x + w]

            # Salva a região recortada como uma nova imagem
            image_count += 1
            output_image_path = os.path.join(output_dir, f"protuberancia_{image_count}.png")
            cv2.imwrite(output_image_path, roi)

            # Anota o tamanho da protuberância
            cv2.putText(original_img_resized, f"{area:.1f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Exibe a imagem original com os círculos desenhados
    plt.imshow(cv2.cvtColor(original_img_resized, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Protuberâncias Detectadas")
    plt.show()

    # Retorna os tamanhos dos calombos e o caminho das imagens salvas
    return lump_sizes, image_count


# Teste com uma nova imagem
test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado3.jpeg'
output_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado_com_calombos'  # Diretório para salvar as imagens processadas

lump_sizes = detect_and_save_protuberances(test_image_path, output_dir)
print("Tamanhos dos calombos detectados:", lump_sizes)
