import keras.backend as K
from keras.layers import Layer
from tensorflow import Tensor


class SparseSoftmax(Layer):
    """Custom layer implementing the sparse softmax activation function."""

    def _init_(self, name="SparseSoftmax", **kwargs):
        super()._init_(name=name, **kwargs)

    def call(self, inputs: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        e_x = K.exp(inputs - K.max(inputs, axis=-1, keepdims=True))
        sum_e_x = K.sum(e_x, axis=-1, keepdims=True)
        output = e_x / (sum_e_x + K.epsilon())
        return output
