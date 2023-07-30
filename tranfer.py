import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub

# To Avoid GPU errors
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# # ================================================ #
# #                  Pretrained-Model                #
# # ================================================ #

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0


model = keras.models.load_model("saved_model/")
model.trainable = False

for layer in model.layers:
    assert layer.trainable == False

base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
final_output = layers.Dense(10)(base_outputs)

new_model = keras.Model(inputs=base_inputs, outputs=final_outputs)
print(model.summary())

new_model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

new_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)