#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential,losses,optimizers,datasets


#%%
x = tf.constant([2.,1.,0.1])
layer = layers.Softmax(axis=-1)
layer(x)
#%%
def proprocess(x,y):
    x = tf.reshape(x, [-1]) 
    return x,y

# x: [60k, 28, 28],
# y: [60k]
(x, y), (x_test,y_test) = datasets.mnist.load_data()
# x: [0~255] => [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32) 

# x: [0~255] => [0~1.]
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32) 

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(proprocess).batch(128)

val_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
val_db = val_db.shuffle(1000).map(proprocess).batch(128)

x,y = next(iter(train_db))
print(x.shape, y.shape)
#%%

from tensorflow.keras import layers, Sequential
network = Sequential([
    layers.Dense(3, activation=None),
    layers.ReLU(),
    layers.Dense(2, activation=None),
    layers.ReLU()
])
x = tf.random.normal([4,3])
network(x)

#%%
layers_num = 2
network = Sequential([])
for _ in range(layers_num):
    network.add(layers.Dense(3))
    network.add(layers.ReLU())
network.build(input_shape=(None, 4))
network.summary()

#%%
for p in network.trainable_variables:
    print(p.name, p.shape)

#%%
# 创建5层的全连接层网络
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(4, 28*28))
network.summary()


#%%
# 导入优化器，损失函数模块
from tensorflow.keras import optimizers,losses 
# 采用Adam优化器，学习率为0.01;采用交叉熵损失函数，包含Softmax
network.compile(optimizer=optimizers.Adam(lr=0.01),
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'] # 设置测量指标为准确率
)


#%%
# 指定训练集为db，验证集为val_db,训练5个epochs，每2个epoch验证一次
history = network.fit(train_db, epochs=5, validation_data=val_db, validation_freq=2)


#%%
history.history # 打印训练记录

#%%
# 保存模型参数到文件上
network.save_weights('weights.ckpt')
print('saved weights.')
del network # 删除网络对象
# 重新创建相同的网络结构
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.compile(optimizer=optimizers.Adam(lr=0.01),
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    ) 
# 从参数文件中读取数据并写入当前网络
network.load_weights('weights.ckpt')
print('loaded weights!')


#%%
# 新建池化层
global_average_layer = layers.GlobalAveragePooling2D()
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4,7,7,2048])
out = global_average_layer(x) # 池化层降维
print(out.shape)


#%%
# 新建全连接层
fc = layers.Dense(100)
# 利用上一层的输出作为本层的输入，测试其输出
x = tf.random.normal([4,2048])
out = fc(x)
print(out.shape)


#%%
