import qkeras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Conv2D
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qconv2d_batchnorm import QConv2DBatchnorm 
from qkeras.quantizers import quantized_bits, quantized_relu
from fkeras.fdense import FQDense

NUM_CLASSES = 10

def dense_mnist():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    return model


def quantized_dense_mnist(
    logit_total_bits, 
    logit_int_bits, 
    activation_total_bits, 
    activation_int_bits, 
    logit_quantizer="quantized_bits",
    activation_quantizer="quantized_relu",
    alpha=1.0, 
    use_stochastic_rounding=True
):
    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(
        logit_total_bits, 
        logit_int_bits, 
        alpha=alpha, 
        use_stochastic_rounding=use_stochastic_rounding
    )
    if activation_quantizer == 'binary_tanh':
        activation_quantizer = qkeras.quantizers.binary_tanh
    else:
        activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(
            activation_total_bits, 
            activation_int_bits, 
            use_stochastic_rounding=use_stochastic_rounding
        )

    input = Input(shape=(28, 28))
    x = QActivation("quantized_bits(8, 0)")(input) 
    x = Flatten()(input)
    x = QDense(128, kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer)(x)
    x = QActivation(activation_quantizer)(x)
    output = QDense(NUM_CLASSES, kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer)(x)
    return Model(inputs=input, outputs=output)

def faulty_quantized_dense_mnist(
    logit_total_bits, 
    logit_int_bits, 
    activation_total_bits, 
    activation_int_bits, 
    logit_quantizer="quantized_bits",
    activation_quantizer="quantized_relu",
    alpha=1.0, 
    use_stochastic_rounding=True
):
    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(
        logit_total_bits, 
        logit_int_bits, 
        alpha=alpha, 
        use_stochastic_rounding=use_stochastic_rounding
    )
    if activation_quantizer == 'binary_tanh':
        activation_quantizer = qkeras.quantizers.binary_tanh
    else:
        activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(
            activation_total_bits, 
            activation_int_bits, 
            use_stochastic_rounding=use_stochastic_rounding
        )

    input = Input(shape=(28, 28))
    x = QActivation("quantized_bits(8, 0)")(input) 
    x = Flatten()(input)
    x = FQDense(128, kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer)(x)
    x = QActivation(activation_quantizer)(x)
    output = FQDense(NUM_CLASSES, kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer)(x)
    return Model(inputs=input, outputs=output)

def conv2d_mnist():
    """
    Taken from: https://keras.io/examples/vision/mnist_convnet/
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    return model
