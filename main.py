import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

train_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\train'
validation_dir = 'C:\\Users\\rober\\Desktop\\PrickTest\\validation'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(150, 150),
    batch_size=32,
    label_mode='binary'
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

model.save('prick_test_model.h5')

def predict_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    if prediction[0] > 0.5:
        return "Positive reaction detected"
    else:
        return "Negative reaction detected"

#positivo
test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado1.png'
#negativo
# test_image_path = 'C:\\Users\\rober\\Desktop\\PrickTest\\resultado\\resultado2.jpeg'
print(predict_image(test_image_path))
