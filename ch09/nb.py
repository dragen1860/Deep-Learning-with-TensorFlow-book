#%%
import tensorflow as tf 
from    tensorflow.keras import layers

pip install -U scikit-learn

#%%
# 添加dropout操作
x = tf.nn.dropout(x, rate=0.5)
# 添加Dropout层
model.add(layers.Dropout(rate=0.5))

# 手动计算每个张量的范数
loss_reg = lambda_ * tf.reduce_sum(tf.square(w))
# 在层方式时添加范数函数
Dense(256, activation='relu',
                    kernel_regularizer=regularizers.l2(_lambda))

#%%
#                     
# 创建网络参数w1,w2
w1 = tf.random.normal([4,3])
w2 = tf.random.normal([4,2])
# 计算L1正则化项
loss_reg = tf.reduce_sum(tf.math.abs(w1))\
    + tf.reduce_sum(tf.math.abs(w2))


# 计算L2正则化项
loss_reg = tf.reduce_sum(tf.square(w1))\
    + tf.reduce_sum(tf.square(w2))

#%%
loss_reg

#%%
