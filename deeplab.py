from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
import tensorflow as tf


def SepConv_BN(x, filters, stride=1, kernel=3, rate=1, epsilon=1e-5):
    x = DepthwiseConv2D(kernel, strides=stride, padding='same', dilation_rate=rate)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    x = Activation(tf.nn.relu)(x)

    x = Conv2D(filters, 1, padding='same', strides=1)(x)
    x = BatchNormalization(epsilon=epsilon)(x)
    x = Activation(tf.nn.relu)(x)
    return x


def xception_block(inputs, filters, str=None, concatenate=True):
    if str is None:
        str = [1, 1, 2]
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual, filters[i], stride=str[i])

    if concatenate:
        inputs = Conv2D(filters[2], 1, padding='same', strides=2)(inputs)
    else:
        inputs = Conv2D(filters[2], 1, padding='same', strides=1)(inputs)
    return Concatenate()([residual, inputs])


def xception(inputs):
    x = Conv2D(32, 3, strides=2)(inputs)
    x = Conv2D(64, 3)(x)
    mid = xception_block(x, [128, 128, 128])
    x = xception_block(mid, [256, 256, 256])
    x = xception_block(x, [728, 728, 728])
    for i in range(16):
        x = xception_block(x, [728, 728, 728], [1, 1, 1], False)
    x = xception_block(x, [728, 1024, 1024], [1, 1, 1], False)
    x = SepConv_BN(x, 1536, rate=2)
    x = SepConv_BN(x, 1536, rate=2)
    x = SepConv_BN(x, 2048, rate=2)
    return x, mid


def DCNN(inputs):
    b4 = GlobalAveragePooling2D()(inputs)
    b4 = Lambda(lambda inputs: K.expand_dims(inputs, 1))(b4)
    b4 = Lambda(lambda inputs: K.expand_dims(inputs, 1))(b4)
    b4 = Conv2D(256, 1, padding='same')(b4)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation(tf.nn.relu)(b4)
    size_before = tf.keras.backend.int_shape(inputs)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3], method='bilinear', align_corners=True))(b4)

    b0 = Conv2D(256, 1, padding='same')(inputs)
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation(tf.nn.relu)(b0)

    b1 = SepConv_BN(inputs, 256, rate=6)
    b2 = SepConv_BN(inputs, 256, rate=12)
    b3 = SepConv_BN(inputs, 256, rate=18)

    outputs = Concatenate()([b4, b0, b1, b2, b3])
    outputs = Conv2D(256, 1, padding='same', strides=1)(outputs)
    outputs = BatchNormalization(epsilon=1e-5)(outputs)
    outputs = Activation(tf.nn.relu)(outputs)
    outputs = Dropout(0.1)(outputs)

    return outputs


def model(inputs_size, classes=1):
    inputs = Input(inputs_size)
    x, mid = xception(inputs)
    x = DCNN(x)
    mid_size = tf.keras.backend.int_shape(mid)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, mid_size[1:3], method='bilinear', align_corners=True))(x)
    dec_mid = Conv2D(48, 1, padding='same')(mid)
    dec_mid = BatchNormalization(epsilon=1e-5)(dec_mid)
    dec_mid = Activation(tf.nn.relu)(dec_mid)
    x = Concatenate()([x, dec_mid])
    x = SepConv_BN(x, 256)
    x = SepConv_BN(x, 256)
    x = Conv2D(classes, 1, padding='same')(x)
    size_inp = tf.keras.backend.int_shape(inputs)
    x = Lambda(lambda xx: tf.compat.v1.image.resize(xx, size_inp[1: 3], method='bilinear', align_corners=True))(x)
    outputs = tf.keras.layers.Activation(tf.nn.sigmoid)(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model
