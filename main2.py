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

def detect_and_measure(img_path):
    # Caminho para salvar a imagem processada
    output_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado3'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, 'result_image.jpg')

    # Carrega e prepara a imagem
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Faz a predição de segmentação
    prediction = model.predict(img_array)
    mask = (prediction[0, :, :, 0] > 0.2).astype(np.uint8)  # Máscara binária

    # Exibir a máscara para debug
    plt.imshow(mask, cmap='gray')
    plt.title("Máscara Gerada")
    plt.axis('off')
    plt.show()

    # Remova as operações de morfologia para debug
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Redimensiona a imagem original para desenhar sobre ela
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (150, 150))

    # Encontra contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lista para armazenar tamanhos dos calombos
    lump_sizes = []

    # Desabilite o filtro de área para debug
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:  # Temporariamente pega todos os contornos
            lump_sizes.append(area)

            # Calcula o centro e o raio do círculo mínimo que envolve o contorno
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Desenha o círculo ao redor do calombo
            cv2.circle(original_img, center, radius, (0, 255, 0), 2)

            # Anota o tamanho do calombo na imagem
            cv2.putText(original_img, f"{area:.1f}", (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Salva a imagem anotada
    cv2.imwrite(save_path, original_img)

    # Exibe a imagem anotada
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Retorna os tamanhos dos calombos
    return lump_sizes


# Teste com uma nova imagem
test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado1.png'
lump_sizes = detect_and_measure(test_image_path)
print("Tamanhos dos calombos detectados:", lump_sizes)
