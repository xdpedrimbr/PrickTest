import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt

train_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\train'
validation_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\validation'

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

train_dataset = train_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x))) 
validation_dataset = validation_dataset.map(lambda x: (x / 255.0, tf.image.rgb_to_grayscale(x)))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2), padding='same'),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),  

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),

    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

model.save('prick_test_segmentation_model.h5')

def detect_and_annotate(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) 

    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (150, 150))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2) 

    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado1.png'
detect_and_annotate(test_image_path)
