import numpy as np
import tensorflow as tf


x=tf.constant([2,20,30,3,6])
print(tf.argmax(x).numpy())