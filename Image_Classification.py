import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import pathlib
import tensorflow as tf
from tensorflow import keras
import cv2
import imghdr
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


data_dir = 'data'
data_dir =pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

## Open the Image
apple = list(data_dir.glob('apple/*'))
# app = Image.open(str(apple[0]))
banana = list(data_dir.glob('banana/*'))
# bana = Image.open(str(banana[0]))
orange = list(data_dir.glob('orange/*'))
# ora = Image.open(str(orange[0]))
""" plt.imshow(app, cmap='gray')
plt.show()
plt.imshow(bana, cmap='gray')
plt.show()
plt.imshow(ora, cmap='gray')
plt.show() """

batch_size = 68
img_height = 190
img_width = 190

# Found 733 files belonging to 3 classes.
# Using 587 files for training.
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size) 


# Found 733 files belonging to 3 classes.
# Using 146 files for validation.

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names) # apple, banana, orange


# Visualize the Picture
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# Configure the dataset for performance

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#Standardize the data
normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                       img_width,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),   
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes ,name="outputs")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


epochs = 18
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_dir = 'test'

for image_class in os.listdir(test_dir):
        img = cv2.imread(image_class)


apple_corr = 0
banana_corr = 0
orange_corr = 0
mixed_corr = 0

for filename in os.listdir(test_dir):
        img = cv2.imread(os.path.join("test", filename))
        # img_data = np.array(img)
        resize = tf.image.resize(img, (img_width, img_height))
        resize = resize.numpy().astype('uint8')
        img_array = tf.keras.utils.img_to_array(resize)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(score)
        print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        if class_names[np.argmax(score)] in filename:
                if class_names[np.argmax(score)] == 'apple':
                        apple_corr += 1
                elif class_names[np.argmax(score)] == 'banana':
                        banana_corr += 1
                elif class_names[np.argmax(score)] == 'orange':
                        orange_corr += 1
                elif class_names[np.argmax(score)] == 'mixed':
                        mixed_corr += 1

print("Apple Correct: " + str(apple_corr))
print("Banana Correct: " + str(banana_corr))
print("Orange Correct: " + str(orange_corr))
print("Mixed Correct: " + str(mixed_corr))

Accuracy = (apple_corr + banana_corr + orange_corr + mixed_corr)/60
Accuracy


ch_test_dir = 'CH_test'
test_ds = tf.keras.utils.image_dataset_from_directory(
    ch_test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size)

evaluation = model.evaluate(test_ds)
print(evaluation);

