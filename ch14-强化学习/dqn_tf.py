import collections
import random
import gym,os
import  numpy as np
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses

env = gym.make('CartPole-v1')  # 创建游戏环境
env.seed(1234)
tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

# Hyperparameters
learning_rate = 0.0002
gamma = 0.99
buffer_limit = 50000
batch_size = 32


class ReplayBuffer():
    # 经验回放池
    def __init__(self):
        # 双向队列
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        # 从回放池采样n个5元组
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        # 按类别进行整理
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        # 转换成Tensor
        return tf.constant(s_lst, dtype=tf.float32),\
                      tf.constant(a_lst, dtype=tf.int32), \
                      tf.constant(r_lst, dtype=tf.float32), \
                      tf.constant(s_prime_lst, dtype=tf.float32), \
                      tf.constant(done_mask_lst, dtype=tf.float32)


    def size(self):
        return len(self.buffer)


class Qnet(keras.Model):
    def __init__(self):
        # 创建Q网络，输入为状态向量，输出为动作的Q值
        super(Qnet, self).__init__()
        self.fc1 = layers.Dense(256, kernel_initializer='he_normal')
        self.fc2 = layers.Dense(256, kernel_initializer='he_normal')
        self.fc3 = layers.Dense(2, kernel_initializer='he_normal')

    def call(self, x, training=None):
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, s, epsilon):
        # 送入状态向量，获取策略: [4]
        s = tf.constant(s, dtype=tf.float32)
        # s: [4] => [1,4]
        s = tf.expand_dims(s, axis=0)
        out = self(s)[0]
        coin = random.random()
        # 策略改进：e-贪心方式
        if coin < epsilon:
            # epsilon大的概率随机选取
            return random.randint(0, 1)
        else:  # 选择Q值最大的动作
            return int(tf.argmax(out))


def train(q, q_target, memory, optimizer):
    # 通过Q网络和影子网络来构造贝尔曼方程的误差，
    # 并只更新Q网络，影子网络的更新会滞后Q网络
    huber = losses.Huber()
    for i in range(10):  # 训练10次
        # 从缓冲池采样
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        with tf.GradientTape() as tape:
            # s: [b, 4]
            q_out = q(s)  # 得到Q(s,a)的分布
            # 由于TF的gather_nd与pytorch的gather功能不一样，需要构造
            # gather_nd需要的坐标参数，indices:[b, 2]
            # pi_a = pi.gather(1, a) # pytorch只需要一行即可实现
            indices = tf.expand_dims(tf.range(a.shape[0]), axis=1)
            indices = tf.concat([indices, a], axis=1)
            q_a = tf.gather_nd(q_out, indices) # 动作的概率值, [b]
            q_a = tf.expand_dims(q_a, axis=1) # [b]=> [b,1]
            # 得到Q(s',a)的最大值，它来自影子网络！ [b,4]=>[b,2]=>[b,1]
            max_q_prime = tf.reduce_max(q_target(s_prime),axis=1,keepdims=True)
            # 构造Q(s,a_t)的目标值，来自贝尔曼方程
            target = r + gamma * max_q_prime * done_mask
            # 计算Q(s,a_t)与目标值的误差
            loss = huber(q_a, target)
        # 更新网络，使得Q(s,a_t)估计符合贝尔曼方程
        grads = tape.gradient(loss, q.trainable_variables)
        # for p in grads:
        #     print(tf.norm(p))
        # print(grads)
        optimizer.apply_gradients(zip(grads, q.trainable_variables))


def main():
    env = gym.make('CartPole-v1')  # 创建环境
    q = Qnet()  # 创建Q网络
    q_target = Qnet()  # 创建影子网络
    q.build(input_shape=(2,4))
    q_target.build(input_shape=(2,4))
    for src, dest in zip(q.variables, q_target.variables):
        dest.assign(src) # 影子网络权值来自Q
    memory = ReplayBuffer()  # 创建回放池

    print_interval = 20
    score = 0.0
    optimizer = optimizers.Adam(lr=learning_rate)

    for n_epi in range(10000):  # 训练次数
        # epsilon概率也会8%到1%衰减，越到后面越使用Q值最大的动作
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s = env.reset()  # 复位环境
        for t in range(600):  # 一个回合最大时间戳
            # if n_epi>1000:
            #     env.render()
            # 根据当前Q网络提取策略，并改进策略
            a = q.sample_action(s, epsilon)
            # 使用改进的策略与环境交互
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0  # 结束标志掩码
            # 保存5元组
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime  # 刷新状态
            score += r  # 记录总回报
            if done:  # 回合结束
                break

        if memory.size() > 2000:  # 缓冲池只有大于2000就可以训练
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            for src, dest in zip(q.variables, q_target.variables):
                dest.assign(src)  # 影子网络权值来自Q
            print("# of episode :{}, avg score : {:.1f}, buffer size : {}, " \
                  "epsilon : {:.1f}%" \
                  .format(n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0
    env.close()


if __name__ == '__main__':
    main()