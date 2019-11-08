#%%

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers
from    tensorflow.keras import datasets
import  os


#%% 
x = tf.random.normal([2,28*28])
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
o1 = tf.matmul(x,w1) + b1
o1
#%%
x = tf.random.normal([4,28*28])
fc1 = layers.Dense(256, activation=tf.nn.relu) 
fc2 = layers.Dense(128, activation=tf.nn.relu) 
fc3 = layers.Dense(64, activation=tf.nn.relu) 
fc4 = layers.Dense(10, activation=None) 
h1 = fc1(x)
h2 = fc2(h1)
h3 = fc3(h2)
h4 = fc4(h3)

model = layers.Sequential([
    layers.Dense(256, activation=tf.nn.relu) ,
    layers.Dense(128, activation=tf.nn.relu) ,
    layers.Dense(64, activation=tf.nn.relu) ,
    layers.Dense(10, activation=None) ,
])
out = model(x)

#%%
256*784+256+128*256+128+64*128+64+10*64+10
#%%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# x: [60k, 28, 28],
# y: [60k]
(x, y), _ = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)

print(x.shape, y.shape, x.dtype, y.dtype)
print(tf.reduce_min(x), tf.reduce_max(x))
print(tf.reduce_min(y), tf.reduce_max(y))


train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)


# [b, 784] => [b, 256] => [b, 128] => [b, 10]
# [dim_in, dim_out], [dim_out]
# 隐藏层1张量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
# 隐藏层2张量
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
# 隐藏层3张量
w3 = tf.Variable(tf.random.truncated_normal([128, 64], stddev=0.1))
b3 = tf.Variable(tf.zeros([64]))
# 输出层张量
w4 = tf.Variable(tf.random.truncated_normal([64, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for epoch in range(10): # iterate db for 10
    for step, (x, y) in enumerate(train_db): # for every batch
        # x:[128, 28, 28]
        # y: [128]

        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, [-1, 28*28])

        with tf.GradientTape() as tape: # tf.Variable
            # x: [b, 28*28]
            #  隐藏层1前向计算，[b, 28*28] => [b, 256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # 隐藏层2前向计算，[b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # 隐藏层3前向计算，[b, 128] => [b, 64] 
            h3 = h2@w3 + b3
            h3 = tf.nn.relu(h3)
            # 输出层前向计算，[b, 64] => [b, 10] 
            h4 = h3@w4 + b4
            out = h4

            # compute loss
            # out: [b, 10]
            # y: [b] => [b, 10]
            y_onehot = tf.one_hot(y, depth=10)

            # mse = mean(sum(y-out)^2)
            # [b, 10]
            loss = tf.square(y_onehot - out)
            # mean: scalar
            loss = tf.reduce_mean(loss)

        # compute gradients
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4])
        # print(grads)
        # w1 = w1 - lr * w1_grad
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        w4.assign_sub(lr * grads[6])
        b4.assign_sub(lr * grads[7])


        if step % 100 == 0:
            print(epoch, step, 'loss:', float(loss))




#%%
