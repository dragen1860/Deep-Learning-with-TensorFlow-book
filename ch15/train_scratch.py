import  matplotlib
from    matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses
from    tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


from pokemon import  load_pokemon,normalize



def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [244, 244])

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=5)

    return x, y


batchsz = 32
# 创建训练集Datset对象
images, labels, table = load_pokemon('pokemon',mode='train')
db_train = tf.data.Dataset.from_tensor_slices((images, labels))
db_train = db_train.shuffle(1000).map(preprocess).batch(batchsz)
# 创建验证集Datset对象
images2, labels2, table = load_pokemon('pokemon',mode='val')
db_val = tf.data.Dataset.from_tensor_slices((images2, labels2))
db_val = db_val.map(preprocess).batch(batchsz)
# 创建测试集Datset对象
images3, labels3, table = load_pokemon('pokemon',mode='test')
db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
db_test = db_test.map(preprocess).batch(batchsz)

# 加载DenseNet网络模型，并去掉最后一层全连接层，最后一个池化层设置为max pooling
net = keras.applications.DenseNet121(include_top=False, pooling='max')
# 设计为不参与优化，即MobileNet这部分参数固定不动
net.trainable = True
newnet = keras.Sequential([
    net, # 去掉最后一层的DenseNet121
    layers.Dense(1024, activation='relu'), # 追加全连接层
    layers.BatchNormalization(), # 追加BN层
    layers.Dropout(rate=0.5), # 追加Dropout层，防止过拟合
    layers.Dense(5) # 根据宝可梦数据的任务，设置最后一层输出节点数为5
])
newnet.build(input_shape=(4,224,224,3))
newnet.summary()

# 创建Early Stopping类，连续3次不下降则终止
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=3
)

newnet.compile(optimizer=optimizers.Adam(lr=1e-3),
               loss=losses.CategoricalCrossentropy(from_logits=True),
               metrics=['accuracy'])
history  = newnet.fit(db_train, validation_data=db_val, validation_freq=1, epochs=100,
           callbacks=[early_stopping])
history = history.history
print(history.keys())
print(history['val_accuracy'])
print(history['accuracy'])
test_acc = newnet.evaluate(db_test)

plt.figure()
returns = history['val_accuracy']
plt.plot(np.arange(len(returns)), returns, label='验证准确率')
plt.plot(np.arange(len(returns)), returns, 's')
returns = history['accuracy']
plt.plot(np.arange(len(returns)), returns, label='训练准确率')
plt.plot(np.arange(len(returns)), returns, 's')

plt.plot([len(returns)-1],[test_acc[-1]], 'D', label='测试准确率')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.savefig('scratch.svg')