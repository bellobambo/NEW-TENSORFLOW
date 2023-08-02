import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# # To Avoid GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0


class Dense(layers.Layer):
    def __init__(self, units, inputs_dim):
        super(Dense, self).__init__()
        self.units = units
        self.inputs_dim = inputs_dim
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1], self.units),  # Corrected variable name here
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            name='b', shape=(self.units,), initializer='zeros', trainable=True,
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class MyRelu(layers.Layer):
    def __init__(self):
        super(MyRelu, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


class MyModel(keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64, 784)
        self.dense2 = Dense(num_classes, self.dense1.units)  # Pass the correct inputs_dim
        self.relu = MyRelu()

    def call(self, input_tensor):
        x = self.relu(self.dense1(input_tensor))
        return self.dense2(x)


model = MyModel()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
