import  matplotlib
from    matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus']=False

plt.figure()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import  threading
import  gym
import  multiprocessing
import  numpy as np
from    queue import Queue
import  matplotlib.pyplot as plt

import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers,optimizers,losses



tf.random.set_seed(1231)
np.random.seed(1231)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


class ActorCritic(keras.Model):
    # Actor-Critic模型
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.state_size = state_size # 状态向量长度
        self.action_size = action_size # 动作数量
        # 策略网络Actor
        self.dense1 = layers.Dense(128, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        # V网络Critic
        self.dense2 = layers.Dense(128, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        # 获得策略分布Pi(a|s)
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        # 获得v(s)
        v = self.dense2(inputs)
        values = self.values(v)
        return logits, values


def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
    # 统计工具函数
    if global_ep_reward == 0:
        global_ep_reward = episode_reward
    else:
        global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
    print(
        f"{episode} | "
        f"Average Reward: {int(global_ep_reward)} | "
        f"Episode Reward: {int(episode_reward)} | "
        f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
        f"Steps: {num_steps} | "
        f"Worker: {worker_idx}"
    )
    result_queue.put(global_ep_reward) # 保存回报，传给主线程
    return global_ep_reward

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Agent:
    # 智能体，包含了中央参数网络server
    def __init__(self):
        # server优化器，client不需要，直接从server拉取参数
        self.opt = optimizers.Adam(1e-3)
        # 中央模型，类似于参数服务器
        self.server = ActorCritic(4, 2) # 状态向量，动作数量
        self.server(tf.random.normal((2, 4)))
    def train(self):
        res_queue = Queue() # 共享队列
        # 创建各个交互环境
        workers = [Worker(self.server, self.opt, res_queue, i)
                   for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print("Starting worker {}".format(i))
            worker.start()
        # 统计并绘制总回报曲线
        returns = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                returns.append(reward)
            else: # 结束标志
                break
        [w.join() for w in workers] # 等待线程退出 

        print(returns)
        plt.figure()
        plt.plot(np.arange(len(returns)), returns)
        # plt.plot(np.arange(len(moving_average_rewards)), np.array(moving_average_rewards), 's')
        plt.xlabel('回合数')
        plt.ylabel('总回报')
        plt.savefig('a3c-tf-cartpole.svg')


class Worker(threading.Thread): 
    def __init__(self,  server, opt, result_queue, idx):
        super(Worker, self).__init__()
        self.result_queue = result_queue # 共享队列
        self.server = server # 中央模型
        self.opt = opt # 中央优化器
        self.client = ActorCritic(4, 2) # 线程私有网络
        self.worker_idx = idx # 线程id
        self.env = gym.make('CartPole-v1').unwrapped
        self.ep_loss = 0.0

    def run(self): 
        mem = Memory() # 每个worker自己维护一个memory
        for epi_counter in range(500): # 未达到最大回合数
            current_state = self.env.reset() # 复位client游戏状态
            mem.clear()
            ep_reward = 0.
            ep_steps = 0  
            done = False
            while not done:
                # 获得Pi(a|s),未经softmax
                logits, _ = self.client(tf.constant(current_state[None, :],
                                         dtype=tf.float32))
                probs = tf.nn.softmax(logits)
                # 随机采样动作
                action = np.random.choice(2, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action) # 交互 
                ep_reward += reward # 累加奖励
                mem.store(current_state, action, reward) # 记录
                ep_steps += 1 # 计算回合步数
                current_state = new_state # 刷新状态 

                if ep_steps >= 500 or done: # 最长步数500
                    # 计算当前client上的误差
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(done, new_state, mem) 
                    # 计算误差
                    grads = tape.gradient(total_loss, self.client.trainable_weights)
                    # 梯度提交到server，在server上更新梯度
                    self.opt.apply_gradients(zip(grads,
                                                 self.server.trainable_weights))
                    # 从server拉取最新的梯度
                    self.client.set_weights(self.server.get_weights())
                    mem.clear() # 清空Memory 
                    # 统计此回合回报
                    self.result_queue.put(ep_reward)
                    print(self.worker_idx, ep_reward)
                    break
        self.result_queue.put(None) # 结束线程

    def compute_loss(self,
                     done,
                     new_state,
                     memory,
                     gamma=0.99):
        if done:
            reward_sum = 0. # 终止状态的v(终止)=0
        else:
            reward_sum = self.client(tf.constant(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]
        # 统计折扣回报
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        # 获取状态的Pi(a|s)和v(s)
        logits, values = self.client(tf.constant(np.vstack(memory.states),
                                 dtype=tf.float32))
        # 计算advantage = R() - v(s)
        advantage = tf.constant(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - values
        # Critic网络损失
        value_loss = advantage ** 2
        # 策略损失
        policy = tf.nn.softmax(logits)
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=memory.actions, logits=logits)
        # 计算策略网络损失时，并不会计算V网络
        policy_loss = policy_loss * tf.stop_gradient(advantage)
        # Entropy Bonus
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy,
                                                          logits=logits)
        policy_loss = policy_loss - 0.01 * entropy
        # 聚合各个误差
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


if __name__ == '__main__':
    master = Agent()
    master.train()
