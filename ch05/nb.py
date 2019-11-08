#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets
import  os


#%%
a = tf.random.normal([4,35,8]) # 模拟成绩册A
b = tf.random.normal([6,35,8]) # 模拟成绩册B
tf.concat([a,b],axis=0) # 合并成绩册


#%%
x = tf.random.normal([2,784])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1) + b1  #
o1 = tf.nn.relu(o1)

#%%
