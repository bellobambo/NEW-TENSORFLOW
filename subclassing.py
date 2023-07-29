from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# CNN
# x10

class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernal_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernal_size, padding='same')  # Fixed typo here
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        print(x.shape)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

model = keras.Sequential([
    CNNBlock(32),
    CNNBlock(32),
    CNNBlock(32),
    layers.Flatten(),
    layers.Dense(10),
])

class ResBlock(layer.Layer):
    def __init__(self, channels=[32, 64, 128]):
        super(ResBlock, sself).__init__()
        self.cnn1 = CNNBlock(channels[0])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=64, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
