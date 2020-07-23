import threading
import time
import numpy as np
import tensorflow as tf


x=np.arange(40).reshape(2,2,10)
print(tf.shape(x)[:-1])
print(tf.reshape(x,[4,-1,5]))

print(66/2)