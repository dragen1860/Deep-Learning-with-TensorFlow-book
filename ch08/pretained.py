#%%
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

#%%
# 加载预训练网络模型，并去掉最后一层
resnet = keras.applications.ResNet50(weights='imagenet',include_top=False)
resnet.summary()
# 测试网络的输出
x = tf.random.normal([4,224,224,3])
out = resnet(x)
out.shape
#%%
# 新建池化层
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4,7,7,2048])
out = global_average_layer(x)
print(out.shape)
#%%
# 新建全连接层
fc = tf.keras.layers.Dense(100)
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4,2048])
out = fc(x)
print(out.shape)
#%%
# 重新包裹成我们的网络模型
mynet = Sequential([resnet, global_average_layer, fc])
mynet.summary()
#%%
resnet.trainable = False
mynet.summary()

#%%