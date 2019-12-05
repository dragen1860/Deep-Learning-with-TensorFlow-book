#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

#%%
x = tf.range(10)
x = tf.random.shuffle(x)
# 创建共10个单词，每个单词用长度为4的向量表示的层
net = layers.Embedding(10, 4)
out = net(x)

out
#%%
net.embeddings
net.embeddings.trainable
net.trainable = False
#%%
# 从预训练模型中加载词向量表
embed_glove = load_embed('glove.6B.50d.txt')
# 直接利用预训练的词向量表初始化Embedding层
net.set_weights([embed_glove])
#%%
cell = layers.SimpleRNNCell(3)
cell.build(input_shape=(None,4))
cell.trainable_variables


#%%
# 初始化状态向量
h0 = [tf.zeros([4, 64])]
x = tf.random.normal([4, 80, 100])
xt = x[:,0,:]
# 构建输入特征f=100,序列长度s=80,状态长度=64的Cell
cell = layers.SimpleRNNCell(64)
out, h1 = cell(xt, h0) # 前向计算
print(out.shape, h1[0].shape)
print(id(out), id(h1[0])) 


#%%
h = h0
# 在序列长度的维度解开输入，得到xt:[b,f]
for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h) # 前向计算
# 最终输出可以聚合每个时间戳上的输出，也可以只取最后时间戳的输出
out = out

#%%
x = tf.random.normal([4,80,100])
xt = x[:,0,:] # 取第一个时间戳的输入x0
# 构建2个Cell,先cell0,后cell1
cell0 = layers.SimpleRNNCell(64)
cell1 = layers.SimpleRNNCell(64)
h0 = [tf.zeros([4,64])] # cell0的初始状态向量
h1 = [tf.zeros([4,64])] # cell1的初始状态向量

out0, h0 = cell0(xt, h0)
out1, h1 = cell1(out0, h1)


#%%
for xt in tf.unstack(x, axis=1):
    # xtw作为输入，输出为out0
    out0, h0 = cell0(xt, h0)
    # 上一个cell的输出out0作为本cell的输入
    out1, h1 = cell1(out0, h1)


#%%
print(x.shape)
# 保存上一层的所有时间戳上面的输出
middle_sequences = []
# 计算第一层的所有时间戳上的输出，并保存
for xt in tf.unstack(x, axis=1):
    out0, h0 = cell0(xt, h0)
    middle_sequences.append(out0)
# 计算第二层的所有时间戳上的输出
# 如果不是末层，需要保存所有时间戳上面的输出
for xt in middle_sequences:
    out1, h1 = cell1(xt, h1)


#%%
layer = layers.SimpleRNN(64)
x = tf.random.normal([4, 80, 100])
out = layer(x)
out.shape

#%%
layer = layers.SimpleRNN(64,return_sequences=True)
out = layer(x) 
out

#%%
net = keras.Sequential([ # 构建2层RNN网络
# 除最末层外，都需要返回所有时间戳的输出
layers.SimpleRNN(64, return_sequences=True),
layers.SimpleRNN(64),
])
out = net(x)



#%%
W = tf.ones([2,2]) # 任意创建某矩阵
eigenvalues = tf.linalg.eigh(W)[0] # 计算特征值
eigenvalues
#%%
val = [W]
for i in range(10): # 矩阵相乘n次方
    val.append([val[-1]@W])
# 计算L2范数
norm = list(map(lambda x:tf.norm(x).numpy(),val))
plt.plot(range(1,12),norm)
plt.xlabel('n times')
plt.ylabel('L2-norm')
plt.savefig('w_n_times_1.svg')
#%%
W = tf.ones([2,2])*0.4 # 任意创建某矩阵
eigenvalues = tf.linalg.eigh(W)[0] # 计算特征值
print(eigenvalues)
val = [W]
for i in range(10):
    val.append([val[-1]@W])
norm = list(map(lambda x:tf.norm(x).numpy(),val))
plt.plot(range(1,12),norm)
plt.xlabel('n times')
plt.ylabel('L2-norm')
plt.savefig('w_n_times_0.svg')
#%%
a=tf.random.uniform([2,2])
tf.clip_by_value(a,0.4,0.6) # 梯度值裁剪

#%%




#%%
a=tf.random.uniform([2,2]) * 5
# 按范数方式裁剪
b = tf.clip_by_norm(a, 5)
tf.norm(a),tf.norm(b)

#%%
w1=tf.random.normal([3,3]) # 创建梯度张量1
w2=tf.random.normal([3,3]) # 创建梯度张量2
# 计算global norm
global_norm=tf.math.sqrt(tf.norm(w1)**2+tf.norm(w2)**2) 
# 根据global norm和max norm=2裁剪
(ww1,ww2),global_norm=tf.clip_by_global_norm([w1,w2],2)
# 计算裁剪后的张量组的global norm
global_norm2 = tf.math.sqrt(tf.norm(ww1)**2+tf.norm(ww2)**2)
print(global_norm, global_norm2)

#%%
with tf.GradientTape() as tape:
  logits = model(x) # 前向传播
  loss = criteon(y, logits) # 误差计算
# 计算梯度值
grads = tape.gradient(loss, model.trainable_variables)
grads, _ = tf.clip_by_global_norm(grads, 25) # 全局梯度裁剪
# 利用裁剪后的梯度张量更新参数
optimizer.apply_gradients(zip(grads, model.trainable_variables))

#%%
x = tf.random.normal([2,80,100])
xt = x[:,0,:] # 得到一个时间戳的输入
cell = layers.LSTMCell(64) # 创建Cell
# 初始化状态和输出List,[h,c]
state = [tf.zeros([2,64]),tf.zeros([2,64])]
out, state = cell(xt, state) # 前向计算
id(out),id(state[0]),id(state[1])


#%%
net = layers.LSTM(4)
net.build(input_shape=(None,5,3))
net.trainable_variables
#%%

net = layers.GRU(4)
net.build(input_shape=(None,5,3))
net.trainable_variables

#%%
# 初始化状态向量
h = [tf.zeros([2,64])]
cell = layers.GRUCell(64) # 新建GRU Cell
for xt in tf.unstack(x, axis=1):
    out, h = cell(xt, h)
out.shape


#%%
