import qkeras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, BatchNormalization, Conv2D
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qconv2d_batchnorm import QConv2DBatchnorm 

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)

"""

def dense_mnist():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


# def quantized_dense_mnist():
#     logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(logit_total_bits, logit_int_bits, alpha=alpha, use_stochastic_rounding=use_stochastic_rounding)
#     if activation_quantizer == 'binary_tanh':
#         activation_quantizer = qkeras.quantizers.binary_tanh
#     else:
#         activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(activation_total_bits, activation_int_bits, use_stochastic_rounding=use_stochastic_rounding)

#     input = Input(shape=(28, 28))
#     x = Flatten()(input)
#     x = QDense(128, kernel_quantizer=quantized_bits(8), bias_quantizer=quantized_bits(8))(x)


# def dense_bn_mnist():

# def conv2d_mnist():

# def conv2dbn_mnist():