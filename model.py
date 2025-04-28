import tensorflow as tf
import keras
from keras import layers, Model

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected. TensorFlow will use CPU.")


# The decorator "@keras.saving.register_keras_serializable()" allows you to save
# your trained model to a .keras file and then load it from the file for testing.
@keras.saving.register_keras_serializable()
class MyModel(Model):
    """Neural network to classify MNIST images."""

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)

        # Use ReLU
        # Last Layer -> Multi-Class, Multinoulli -> Softmax

        self.flatten = (keras.layers.Flatten(input_shape=(28, 28), name="flatten"))
        self.dense1 = keras.layers.Dense(8, activation="relu")
        #self.dense1 = keras.layers.Dense(8, activation="relu")
        self.out = keras.layers.Dense(10, activation="softmax")
    
    def call(self, x, training=False):
        """
        Forward pass.

        Parameters
        ----------
        x : tensor float32 (28, 28)
            Input MNIST image.
        training : bool, optional
            training=True is only needed if there are layers with different
            behavior during training versus inference (e.g. Dropout).
            The default is False.

        Returns
        -------
        out : tensor float32 (None, 10)
              Class probabilities.

        """

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x
