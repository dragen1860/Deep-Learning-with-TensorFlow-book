#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers

#%%
def sigmoid(x): # sigmoid函数，也可以直接使用tf.nn.sigmoid
    return 1 / (1 + tf.math.exp(-x))

def derivative(x): # sigmoid导数的计算
    return sigmoid(x)*(1-sigmoid(x))
