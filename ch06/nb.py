#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers
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
o1
#%%
x = tf.random.normal([4,28*28])
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(512, activation=tf.nn.relu) 
h1 = fc(x)  # 通过fc类完成一次全连接层的计算


#%%
vars(fc)

#%%
x = tf.random.normal([4,4])
# 创建全连接层，指定输出节点数和激活函数
fc = layers.Dense(3, activation=tf.nn.relu) 
h1 = fc(x)  # 通过fc类完成一次全连接层的计算


#%%
fc.non_trainable_variables

#%%
embedding = layers.Embedding(10000, 100)

#%%
x = tf.ones([25000,80])

#%%

embedding(x)

#%%
z = tf.random.normal([2,10]) # 构造输出层的输出
y_onehot = tf.constant([1,3]) # 构造真实值
y_onehot = tf.one_hot(y_onehot, depth=10) # one-hot编码
# 输出层未使用Softmax函数，故from_logits设置为True
loss = keras.losses.categorical_crossentropy(y_onehot,z,from_logits=True)
loss = tf.reduce_mean(loss) # 计算平均交叉熵损失
loss


#%%
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
loss = criteon(y_onehot,z) # 计算损失
loss


#%%
