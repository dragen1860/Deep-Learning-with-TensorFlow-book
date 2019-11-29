#%%
import  os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets, Sequential



#%% 
x = tf.random.normal([2,5,5,3]) # 模拟输入，3通道，高宽为5
# 需要根据[k,k,cin,cout]格式创建，4个卷积核
w = tf.random.normal([3,3,3,4]) 
# 步长为1, padding为0,
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[0,0],[0,0],[0,0]])


# %%
x = tf.random.normal([2,5,5,3]) # 模拟输入，3通道，高宽为5
# 需要根据[k,k,cin,cout]格式创建，4个卷积核
w = tf.random.normal([3,3,3,4])
# 步长为1, padding为1,
out = tf.nn.conv2d(x,w,strides=1,padding=[[0,0],[1,1],[1,1],[0,0]])


# %%
x = tf.random.normal([2,5,5,3]) # 模拟输入，3通道，高宽为5
w = tf.random.normal([3,3,3,4]) # 4个3x3大小的卷积核
# 步长为,padding设置为输出、输入同大小
# 需要注意的是, padding=same只有在strides=1时才是同大小
out = tf.nn.conv2d(x,w,strides=1,padding='SAME')


# %%
x = tf.random.normal([2,5,5,3])
w = tf.random.normal([3,3,3,4])
# 高宽按3倍减少
out = tf.nn.conv2d(x,w,strides=3,padding='SAME')
print(out.shape)


# %%
# 根据[cout]格式创建偏置向量
b = tf.zeros([4])
# 在卷积输出上叠加偏置向量，它会自动broadcasting为[b,h',w',cout]
out = out + b


# %%
# 创建卷积层类
layer = layers.Conv2D(4,kernel_size=(3,4),strides=(2,1),padding='SAME')
out = layer(x) # 前向计算
out.shape


# %%
layer.kernel,layer.bias
# 返回所有待优化张量列表
layer.trainable_variables

# %%
from tensorflow.keras import Sequential
network = Sequential([ # 网络容器
    layers.Conv2D(6,kernel_size=3,strides=1), # 第一个卷积层, 6个3x3卷积核
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
    layers.ReLU(), # 激活函数
    layers.Conv2D(16,kernel_size=3,strides=1), # 第二个卷积层, 16个3x3卷积核
    layers.MaxPooling2D(pool_size=2,strides=2), # 高宽各减半的池化层
    layers.ReLU(), # 激活函数
    layers.Flatten(), # 打平层，方便全连接层处理

    layers.Dense(120, activation='relu'), # 全连接层，120个节点
    layers.Dense(84, activation='relu'), # 全连接层，84节点
    layers.Dense(10) # 全连接层，10个节点
                    ])
# build一次网络模型，给输入X的形状，其中4为随意给的batchsz
network.build(input_shape=(4, 28, 28, 1))
# 统计网络信息
network.summary()


# %%
# 导入误差计算，优化器模块
from tensorflow.keras import losses, optimizers
# 创建损失函数的类，在实际计算时直接调用类实例即可
criteon = losses.CategoricalCrossentropy(from_logits=True)

# %%
# 构建梯度记录环境
with tf.GradientTape() as tape: 
    # 插入通道维度，=>[b,28,28,1]
    x = tf.expand_dims(x,axis=3)
    # 前向计算，获得10类别的预测分布，[b, 784] => [b, 10]
    out = network(x)
    # 真实标签one-hot编码，[b] => [b, 10]
    y_onehot = tf.one_hot(y, depth=10)
    # 计算交叉熵损失函数，标量
    loss = criteon(y_onehot, out)
# 自动计算梯度
grads = tape.gradient(loss, network.trainable_variables)
# 自动更新参数
optimizer.apply_gradients(zip(grads, network.trainable_variables))


# %%
# 记录预测正确的数量，总样本数量
correct, total = 0,0
for x,y in db_test: # 遍历所有训练集样本
    # 插入通道维度，=>[b,28,28,1]
    x = tf.expand_dims(x,axis=3)
    # 前向计算，获得10类别的预测分布，[b, 784] => [b, 10]
    out = network(x)
    # 真实的流程时先经过softmax，再argmax
    # 但是由于softmax不改变元素的大小相对关系，故省去
    pred = tf.argmax(out, axis=-1)  
    y = tf.cast(y, tf.int64)
    # 统计预测正确数量
    correct += float(tf.reduce_sum(tf.cast(tf.equal(pred, y),tf.float32)))
    # 统计预测样本总数
    total += x.shape[0]
    # 计算准确率
    print('test acc:', correct/total)


# %%
# 构造输入
x=tf.random.normal([100,32,32,3])
# 将其他维度合并，仅保留通道维度
x=tf.reshape(x,[-1,3])
# 计算其他维度的均值
ub=tf.reduce_mean(x,axis=0)
ub


# %%
# 创建BN层
layer=layers.BatchNormalization()

# %%
network = Sequential([ # 网络容器
    layers.Conv2D(6,kernel_size=3,strides=1),
    # 插入BN层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Conv2D(16,kernel_size=3,strides=1),
    # 插入BN层
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2,strides=2),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    # 此处也可以插入BN层
    layers.Dense(84, activation='relu'), 
    # 此处也可以插入BN层
    layers.Dense(10)
                    ])


# %%
with tf.GradientTape() as tape: 
    # 插入通道维度
    x = tf.expand_dims(x,axis=3)
    # 前向计算，设置计算模式，[b, 784] => [b, 10]
    out = network(x, training=True)


# %%
for x,y in db_test: # 遍历测试集
    # 插入通道维度
    x = tf.expand_dims(x,axis=3)
    # 前向计算，测试模式
    out = network(x, training=False)


# %%
def preprocess(x, y):
    # [0~1]
    x = 2*tf.cast(x, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)
    return x,y
    
# 在线下载，加载CIFAR10数据集
(x,y), (x_test, y_test) = datasets.cifar100.load_data()
# 删除y的一个维度，[b,1] => [b]
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
# 打印训练集和测试集的形状
print(x.shape, y.shape, x_test.shape, y_test.shape)
# 构建训练集对象
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
# 构建测试集对象
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)
# 从训练集中采样一个Batch，观察
sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))



# %%
conv_layers = [ # 先创建包含多层的列表
    # Conv-Conv-Pooling单元1
    # 64个3x3卷积核, 输入输出同大小
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    # 高宽减半
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元2,输出通道提升至128，高宽大小减半
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元3,输出通道提升至256，高宽大小减半
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元4,输出通道提升至512，高宽大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

    # Conv-Conv-Pooling单元5,输出通道提升至512，高宽大小减半
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]
# 利用前面创建的层列表构建网络容器
conv_net = Sequential(conv_layers)


# %%
# 创建3层全连接层子网络
fc_net = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(100, activation=None),
])


# %%
# build2个子网络，并打印网络参数信息
conv_net.build(input_shape=[4, 32, 32, 3])
fc_net.build(input_shape=[4, 512])
conv_net.summary()
fc_net.summary()


# %%
# 列表合并，合并2个子网络的参数
variables = conv_net.trainable_variables + fc_net.trainable_variables
# 对所有参数求梯度
grads = tape.gradient(loss, variables)
# 自动更新
optimizer.apply_gradients(zip(grads, variables))


# %%
x = tf.random.normal([1,7,7,1]) # 模拟输入
# 空洞卷积，1个3x3的卷积核
layer = layers.Conv2D(1,kernel_size=3,strides=1,dilation_rate=2)
out = layer(x) # 前向计算
out.shape


# %%
# 创建X矩阵
x = tf.range(25)+1
# Reshape为合法维度的张量
x = tf.reshape(x,[1,5,5,1])
x = tf.cast(x, tf.float32)
# 创建固定内容的卷积核矩阵
w = tf.constant([[-1,2,-3.],[4,-5,6],[-7,8,-9]])
# 调整为合法维度的张量
w = tf.expand_dims(w,axis=2)
w = tf.expand_dims(w,axis=3)
# 进行普通卷积运算
out = tf.nn.conv2d(x,w,strides=2,padding='VALID')
out
#%%
# 普通卷积的输出作为转置卷积的输入，进行转置卷积运算
xx = tf.nn.conv2d_transpose(out, w, strides=2, 
    padding='VALID',
    output_shape=[1,5,5,1])


<tf.Tensor: id=117, shape=(5, 5), dtype=float32, numpy=
array([[   67.,  -134.,   278.,  -154.,   231.],
       [ -268.,   335.,  -710.,   385.,  -462.],
       [  586.,  -770.,  1620.,  -870.,  1074.],
       [ -468.,   585., -1210.,   635.,  -762.],
       [  819.,  -936.,  1942., -1016.,  1143.]], dtype=float32)>


# %%
x = tf.random.normal([1,6,6,1])
# 6x6的输入经过普通卷积
out = tf.nn.conv2d(x,w,strides=2,padding='VALID')
out
<tf.Tensor: id=21, shape=(1, 2, 2, 1), dtype=float32, numpy=
array([[[[ 20.438847 ],
         [ 19.160788 ]],

        [[  0.8098897],
         [-28.30303  ]]]], dtype=float32)>
# %%
# 恢复出6x6大小
xx = tf.nn.conv2d_transpose(out, w, strides=2, 
    padding='VALID',
    output_shape=[1,6,6,1])
xx



# %%
# 创建转置卷积类
layer = layers.Conv2DTranspose(1,kernel_size=3,strides=2,padding='VALID')
xx2 = layer(out)
xx2
# %%
class BasicBlock(layers.Layer):
    # 残差模块类
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # f(x)包含了2个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1: # 插入identity层
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else: # 否则，直接连接
            self.downsample = lambda x:x


    def call(self, inputs, training=None):
        # 前向传播函数
        out = self.conv1(inputs) # 通过第一个卷积层
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out) # 通过第二个卷积层
        out = self.bn2(out)
        # 输入通过identity()转换
        identity = self.downsample(inputs)
        # f(x)+x运算
        output = layers.add([out, identity])
        # 再通过激活函数并返回
        output = tf.nn.relu(output)
        return output
