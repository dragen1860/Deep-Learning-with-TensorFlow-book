import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers


class Generator(keras.Model):
    # 生成器网络
    def __init__(self):
        super(Generator, self).__init__()
        filter = 64
        # 转置卷积层1,输出channel为filter*8,核大小4,步长1,不使用padding,不使用偏置
        self.conv1 = layers.Conv2DTranspose(filter*8, 4,1, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 转置卷积层2
        self.conv2 = layers.Conv2DTranspose(filter*4, 4,2, 'same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 转置卷积层3
        self.conv3 = layers.Conv2DTranspose(filter*2, 4,2, 'same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 转置卷积层4
        self.conv4 = layers.Conv2DTranspose(filter*1, 4,2, 'same', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 转置卷积层5
        self.conv5 = layers.Conv2DTranspose(3, 4,2, 'same', use_bias=False)

    def call(self, inputs, training=None):
        x = inputs # [z, 100]
        # Reshape乘4D张量，方便后续转置卷积运算:(b, 1, 1, 100)
        x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
        x = tf.nn.relu(x) # 激活函数
        # 转置卷积-BN-激活函数:(b, 4, 4, 512)
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # 转置卷积-BN-激活函数:(b, 8, 8, 256)
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # 转置卷积-BN-激活函数:(b, 16, 16, 128)
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # 转置卷积-BN-激活函数:(b, 32, 32, 64)
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # 转置卷积-激活函数:(b, 64, 64, 3)
        x = self.conv5(x)
        x = tf.tanh(x) # 输出x范围-1~1,与预处理一致

        return x


class Discriminator(keras.Model):
    # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()
        filter = 64
        # 卷积层
        self.conv1 = layers.Conv2D(filter, 4, 2, 'valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        # 卷积层
        self.conv2 = layers.Conv2D(filter*2, 4, 2, 'valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        # 卷积层
        self.conv3 = layers.Conv2D(filter*4, 4, 2, 'valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        # 卷积层
        self.conv4 = layers.Conv2D(filter*8, 3, 1, 'valid', use_bias=False)
        self.bn4 = layers.BatchNormalization()
        # 卷积层
        self.conv5 = layers.Conv2D(filter*16, 3, 1, 'valid', use_bias=False)
        self.bn5 = layers.BatchNormalization()
        # 全局池化层
        self.pool = layers.GlobalAveragePooling2D()
        # 特征打平
        self.flatten = layers.Flatten()
        # 2分类全连接层
        self.fc = layers.Dense(1)


    def call(self, inputs, training=None):
        # 卷积-BN-激活函数:(4, 31, 31, 64)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
        # 卷积-BN-激活函数:(4, 14, 14, 128)
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # 卷积-BN-激活函数:(4, 6, 6, 256)
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        # 卷积-BN-激活函数:(4, 4, 4, 512)
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # 卷积-BN-激活函数:(4, 2, 2, 1024)
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # 卷积-BN-激活函数:(4, 1024)
        x = self.pool(x)
        # 打平
        x = self.flatten(x)
        # 输出，[b, 1024] => [b, 1]
        logits = self.fc(x)

        return logits

def main():

    d = Discriminator()
    g = Generator()


    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)




if __name__ == '__main__':
    main()