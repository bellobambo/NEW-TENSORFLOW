# Imports needed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

img_height = 28
img_width = 28
batch_size = 2

model = keras.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(16, 3, padding="same"),
    layers.Conv2D(32, 3, padding="same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    "data/mnist_subfolders/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    "data/mnist_subfolders/",
    labels="inferred",
    label_mode="int",  # categorical, binary
    # class_names=['0', '1', '2', '3', ...]
    color_mode="grayscale",
    batch_size=batch_size,
    image_size=(img_height, img_width),  # reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation",
)


def augment(x, y):
    image = tf.image.random_brightness(x, max_delta=0.05)
    return image, y


ds_train = ds_train.map(augment)

for epochs in range(10):
    for x, y in ds_train:
        pass

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)


model.fit(ds_train, epochs=10, verbose=2)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=(0.99, 0.99),
    horizontail_flip=False,
    vertical_flip=False,
    data_format='channels_last',
    validation_split=0.0,
    dtype=tf.float32,

)

train_generatior = datagen.flow_from_directory(
    'data/mnist_subfolders/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='sparse',
    shuffle=True,
    subset='training',
    seed=123,
)

def training(): pass

for epoch in range(10):
    num_batches = 0

    for x,y in ds_train:
        num_batches += 1

        training()

        if num_batches == 25:
            break


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"],
)

# using model.fit (note steps_per_epoch)
model.fit(
    epochs=10,
    steps_per_epoch=25,
    verbose=2,
    # if we had a validation generator:
    # validation_data=validation_generator,
    # valiation_steps=len(validation_set)/batch_size),
)