import numpy as np
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *
import tensorflow as tf


inputs = np.random.random((2, 3, 4, 3))
print(inputs)
b4 = GlobalAveragePooling2D()(inputs)
print(b4)
b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
print(b4)
b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
print(b4)

a = np.array([[[[0.38273773, 0.46700382, 0.62068826]]], [[[0.5337202, 0.48855972, 0.49095526]]]])
print(a.shape)
