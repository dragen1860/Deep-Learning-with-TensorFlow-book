import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers


# 2 images with 4x4 size, 3 channels
# we explicitly enforce the mean and stddev to N(1, 0.5)
x = tf.random.normal([2,4,4,3], mean=1.,stddev=0.5)

net = layers.BatchNormalization(axis=-1, center=True, scale=True,
                                trainable=True)

out = net(x)
print('forward in test mode:', net.variables)


out = net(x, training=True)
print('forward in train mode(1 step):', net.variables)

for i in range(100):
    out = net(x, training=True)
print('forward in train mode(100 steps):', net.variables)


optimizer = optimizers.SGD(lr=1e-2)
for i in range(10):
    with tf.GradientTape() as tape:
        out = net(x, training=True)
        loss = tf.reduce_mean(tf.pow(out,2)) - 1

    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
print('backward(10 steps):', net.variables)




