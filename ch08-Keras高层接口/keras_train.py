import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras



def preprocess(x, y):
    # [0~255] => [-1~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1.
    y = tf.cast(y, dtype=tf.int32)
    return x,y


batchsz = 128
# [50k, 32, 32, 3], [10k, 1]
(x, y), (x_val, y_val) = datasets.cifar10.load_data()
y = tf.squeeze(y)
y_val = tf.squeeze(y_val)
y = tf.one_hot(y, depth=10) # [50k, 10]
y_val = tf.one_hot(y_val, depth=10) # [10k, 10]
print('datasets:', x.shape, y.shape, x_val.shape, y_val.shape, x.min(), x.max())


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.map(preprocess).shuffle(10000).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test_db = test_db.map(preprocess).batch(batchsz)


sample = next(iter(train_db))
print('batch:', sample[0].shape, sample[1].shape)


class MyDense(layers.Layer):
    # to replace standard layers.Dense()
    def __init__(self, inp_dim, outp_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [inp_dim, outp_dim])
        # self.bias = self.add_variable('b', [outp_dim])

    def call(self, inputs, training=None):

        x = inputs @ self.kernel
        return x

class MyNetwork(keras.Model):

    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)



    def call(self, inputs, training=None):
        """

        :param inputs: [b, 32, 32, 3]
        :param training:
        :return:
        """
        x = tf.reshape(inputs, [-1, 32*32*3])
        # [b, 32*32*3] => [b, 256]
        x = self.fc1(x)
        x = tf.nn.relu(x)
        # [b, 256] => [b, 128]
        x = self.fc2(x)
        x = tf.nn.relu(x)
        # [b, 128] => [b, 64]
        x = self.fc3(x)
        x = tf.nn.relu(x)
        # [b, 64] => [b, 32]
        x = self.fc4(x)
        x = tf.nn.relu(x)
        # [b, 32] => [b, 10]
        x = self.fc5(x)

        return x


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

network.evaluate(test_db)
network.save_weights('ckpt/weights.ckpt')
del network
print('saved to ckpt/weights.ckpt')


network = MyNetwork()
network.compile(optimizer=optimizers.Adam(lr=1e-3),
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
network.load_weights('ckpt/weights.ckpt')
print('loaded weights from file.')
network.evaluate(test_db)