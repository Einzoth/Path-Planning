import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import einops
# np.random.seed(1)
# torch.manual_seed(1)


BATCH_SIZE = 128            #批大小，每次从记忆库中提取多少个样本进行训练
LR = 0.01                   #learning rate
INITIAL_EPSILON = 0.5       #最优选择动作百分比
FINAL_EPSILON = 0.01
GAMMA = 0.9                 #奖励递减参数
TARGET_REPLACE_ITER = 200   #Q现实网络的更新频率
MEMORY_CAPACITY = 80000     #记忆库大小
SIZE = 9

class SumTree(object):
	
	data_number = 0

	def __init__(self,capacity):
		self.capacity = capacity
		self.tree    = np.zeros(2 * capacity - 1)
		self.data_s  = np.zeros([capacity, 4, SIZE, SIZE], dtype=object)
		self.data_s_ = np.zeros([capacity, 4, SIZE, SIZE], dtype=object)
		self.data_r  = np.zeros(capacity, dtype=object)
		self.data_a  = np.zeros(capacity, dtype=object)
		self.data_p  = np.zeros(capacity, dtype=object)

	def add(self, p, s, a, r, s_, pro):
		tree_idx = self.data_number + self.capacity - 1
		self.data_s [self.data_number] = s
		self.data_s_[self.data_number] = s_
		self.data_a [self.data_number] = a
		self.data_r [self.data_number] = r
		self.data_p [self.data_number] = pro
		self.update(tree_idx, p)
		self.data_number += 1
		if self.data_number >= self.capacity:
			self.data_number = 0

	def update(self, tree_idx, p):
		change = p - self.tree[tree_idx]
		self.tree[tree_idx] = p
		while tree_idx != 0:
			tree_idx = (tree_idx - 1) // 2
			self.tree[tree_idx] += change

	def get_leaf(self,v):
		parent_idx = 0
		while True:
			left_idx = 2 * parent_idx + 1
			right_idx = left_idx + 1
			if left_idx >= len(self.tree):
				leaf_idx = parent_idx
				break
			else:
				if v <= self.tree[left_idx]:
					parent_idx = left_idx
				else:
					v -= self.tree[left_idx]
					parent_idx = right_idx
		
		data_idx = leaf_idx - self.capacity + 1
		return leaf_idx, self.tree[leaf_idx], self.data_s[data_idx], self.data_a[data_idx], self.data_r[data_idx], self.data_s_[data_idx], self.data_p[data_idx]
	
	@property
	def total_p(self):
		return self.tree[0]

class Memory(object):
	epsilon = 0.01
	alpha = 0.6
	beta = 0.4
	beta_increment_per_sampling = 0.001
	abs_err_upper = 1.

	def __init__(self, capacity):
		self.tree = SumTree(capacity)
		self.prio_max = 0.1

	def store(self, s, a, r, s_, pro):
		max_p = np.max(self.tree.tree[-self.tree.capacity:])

		if max_p == 0:
			max_p = self.abs_err_upper
		self.tree.add(max_p, s, a, r, s_, pro)

		max_p = (np.abs(self.prio_max) + self.epsilon) ** self.alpha
		self.tree.add(max_p, s, a, r, s_, pro)

	def sample(self, n):
		b_idx = np.zeros((n,), dtype=np.int32)
		b_s = np.zeros([n, 4, SIZE, SIZE])
		b_a = np.zeros([n, 1])
		b_r = np.zeros([n, 1])
		b_s_ = np.zeros([n, 4, SIZE, SIZE])
		b_p = np.zeros([n, 1])
		ISWeights = np.zeros([n, 1])
		pri_seg = self.tree.total_p / n
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

		min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
		for i in range(n):
			a,b = pri_seg * i, pri_seg * (i+1)
			v = np.random.uniform(a,b)
			idx,p,data_s,data_a,data_r,data_s_,data_p = self.tree.get_leaf(v)
			prob = p / self.tree.total_p

			ISWeights[i,0] = np.power(prob/min_prob, -self.beta)

			b_idx[i],b_s[i],b_a[i],b_r[i],b_s_[i], b_p[i] = idx, data_s, data_a, data_r, data_s_, data_p
		return b_idx,b_s,b_a,b_r,b_s_,b_p,ISWeights
	
	def batch_update(self, tree_idx, abs_errors):
		abs_errors += self.epsilon
		clipped_errors = np.minimum(abs_errors.data, self.abs_err_upper)
		ps = np.power(clipped_errors, self.alpha)
		for ti,p in zip(tree_idx, ps):
			self.tree.update(ti, p)

class Net(nn.Module):
	def __init__(self, n_feature, n_hidden, n_actions):
		super(Net, self).__init__()
		# self.conv1 = nn.Conv2d(
		# 	in_channels=4,
		# 	out_channels=4,
		# 	kernel_size=5,
		# 	stride=1,
		# 	padding=2
		# )
		# self.conv2 = nn.Conv2d(
		# 	in_channels=4,
		# 	out_channels=4,
		# 	kernel_size=3,
		# 	stride=1,
		# 	padding=1
		# )
		# self.conv3 = nn.Conv2d(
		# 	in_channels=4,
		# 	out_channels=4,
		# 	kernel_size=3,
		# 	stride=1,
		# 	padding=1
		# )

		self.lstm = nn.LSTM(
			input_size = 36,
			hidden_size = 36,
			num_layers=4
		)

		self.fc1 = nn.Linear(n_feature * 4, n_hidden)
		self.fc2 = nn.Linear(n_hidden, 64)
		self.values = nn.Linear(64,1)
		self.advantage = nn.Linear(64,n_actions)

	def forward(self, x):
		# x = F.relu(self.conv1(x))
		# x = F.relu(self.conv2(x))
		# x = F.relu(self.conv3(x))
		x = einops.rearrange(x, 'b c h w -> h b (c w)')
		x, (h, c) = self.lstm(x)
		x = x.view(x.size(1), -1)
		# print(x)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		value = self.values(x)
		advantages = self.advantage(x)
		actions_value = value + (advantages - torch.mean(advantages, dim=1, keepdim=True))
		# actions_value = x[-1]
		return actions_value

class DeepQNetwork():
	def __init__(self, n_actions, n_features, n_hidden = 256,
				):
		self.n_actions = n_actions
		self.n_features = n_features
		self.n_hidden = n_hidden
		self.epsilon = INITIAL_EPSILON
		self.eval_net = Net(self.n_features, self.n_hidden, self.n_actions)
		self.target_net = Net(self.n_features, self.n_hidden, self.n_actions)
		
		self.optimizer = optim.SGD(self.eval_net.parameters(), lr=LR)
		self.learn_step_counter = 0
		self.timestep = 0
		self.memory = Memory(capacity=MEMORY_CAPACITY)
		self.cost_his = []

	def store_transition(self, s, a, r, s_, pro):
		s = torch.Tensor(s)
		s_ = torch.Tensor(s_)
		self.memory.store(s, a, r, s_, pro)

	# def choose_action(self, state_current, danger):
	def choose_action(self, state_current):
		state_current = torch.Tensor(state_current)
		state_current = state_current.unsqueeze(0)
		actions_value = self.eval_net(state_current)
		# danger = torch.Tensor(danger)
		# danger会影响神经网络的学习
		# actions_value = actions_value + danger
		if np.random.uniform() >= self.epsilon:
			action = torch.max(actions_value, dim=1)[1]
		else:
			action = np.random.randint(0, self.n_actions)
		self.timestep += 1
		if self.epsilon > FINAL_EPSILON and self.timestep > MEMORY_CAPACITY:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / MEMORY_CAPACITY
		
		return action
	
	def back_choose_action(self, state_current, action_p):
		#对观测值的维度进行扩张
		state_current = torch.Tensor(state_current)
		state_current = state_current.unsqueeze(0)
		action_back = [0,1,2,3]
		action_back.remove(action_p)
		if np.random.uniform() >= self.epsilon:
			#90%会选择Q值最大的动作
			actions_value = self.eval_net(state_current)
			action = torch.max(actions_value, dim=1)[1]
			if action == action_p:
				action = np.random.choice(action_back)
		else:
			#10%会在所有动作随机选择一个
			action = np.random.choice(action_back)

		return action

	def back2_choose_action(self, state_current, action_p, action_p2):
		#对观测值的维度进行扩张
		state_current = torch.Tensor(state_current)
		state_current = state_current.unsqueeze(0)
		action_back = [0,1,2,3]
		action_back.remove(action_p)
		action_back.remove(action_p2)
		if np.random.uniform() >= self.epsilon:
			#90%会选择Q值最大的动作
			actions_value = self.eval_net(state_current)
			action = torch.max(actions_value, dim=1)[1]
			if action == action_p or action == action_p2:
				action = np.random.choice(action_back)
		else:
			#10%会在所有动作随机选择一个
			action = np.random.choice(action_back)

		return action

	def learn(self):
		if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
			self.target_net.load_state_dict(self.eval_net.state_dict())
		tree_idx, batch_s, batch_a, batch_r, batch_s_, batch_p, ISWeights = self.memory.sample(BATCH_SIZE)

		# print(batch_s)
		batch_s = torch.Tensor(batch_s)
		batch_s_ = torch.Tensor(batch_s_)

		q_eval = self.eval_net(batch_s)
		q_next = self.target_net(batch_s_)
		# q_next = self.target_net(batch_s_).detach()
		q_target = torch.Tensor(q_eval.data.numpy().copy())


		batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
		eval_act_index = batch_a.astype(int)
		reward = torch.Tensor(batch_r)
		selected_q_next = torch.max(q_next, dim=1)[0]
		q_target[batch_index, eval_act_index] = reward + GAMMA * selected_q_next

		# self.abs_errors = torch.sum(torch.abs(q_target - q_eval), dim=1)
		self.abs_errors = torch.Tensor(batch_p)


		self.memory.batch_update(tree_idx, self.abs_errors)
		criterion = nn.SmoothL1Loss()
		loss = torch.mean((criterion(q_eval, q_target) * torch.Tensor(ISWeights)))
		# loss = torch.mean((criterion(q_eval, q_target) ))
		# print(ISWeights)

		self.cost_his.append(loss.detach().numpy())
		self.learn_step_counter += 1
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Lost')
		plt.xlabel('training steps')
		plt.show()