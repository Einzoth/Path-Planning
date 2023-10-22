from env import Maze
from RL_brain import DeepQNetwork
import time
import numpy as np

def run_maze():
	#设置总计算步数和单次计算步数，积累学习素材
	step = 0
	loop = 0
	loop_min = 250
	# path = np.zeros((100,2))
	# path_action = np.zeros((100,1))
	# al = [2,2,2,1,1,1,1,1,1,3,3,3,2,2,2,0,0,0,0,0,0,3,3,3]
	# all = 0
	cs = 0
	x_b = [0, 0, 0, 0]
	y_b = [0, 0, 0, 0]

	coverage_max = 0
	start_time = time.time()  
	for episode in range(6000):
		print("episode: {}".format(episode))
		observation = env.reset()
		back = False
		observation_p0, observation_p1, observation_p2, observation_p3 = process(observation)
		# all = 0
		state_current0 = np.stack((observation_p0, observation_p0, observation_p0, observation_p0), axis=0)
		state_current1 = np.stack((observation_p1, observation_p1, observation_p1, observation_p1), axis=0)
		state_current2 = np.stack((observation_p2, observation_p2, observation_p2, observation_p2), axis=0)
		state_current3 = np.stack((observation_p3, observation_p3, observation_p3, observation_p3), axis=0)
		while True:
			# print("step: {}".format(step))
			# danger = env.render()
			#根据观测状态选择动作
			action0 = RL.choose_action(state_current0)
			action1 = RL.choose_action(state_current1)
			action2 = RL.choose_action(state_current2)
			action3 = RL.choose_action(state_current3)
			# obs_action = al[all]
			# all += 1
			# if all == 24:
				# all = 0
			state, reward, done, x, y, coverage, out, action_back = env.step(action0, action1, action2, action3, x_b, y_b, back)
			x_b = x
			y_b = y
			# state_back_y = y
			# obs_x_ = obs_x
			# obs_y_ = obs_y
			# state_new0 = np.stack((state_current0[1], state_current0[2], state_current0[3], state[0]), axis = 0)
			# state_new1 = np.stack((state_current1[1], state_current1[2], state_current1[3], state[1]), axis = 0)
			# state_new2 = np.stack((state_current2[1], state_current2[2], state_current2[3], state[2]), axis = 0)
			# state_new3 = np.stack((state_current3[1], state_current3[2], state_current3[3], state[3]), axis = 0)
			#回溯算法
			if out:
				back = True
				i = 0
				j = 0
				while back:
					if action_back == 0:
						action0 = RL.back_choose_action(state_current0,action0)
					elif action_back == 1:
						action1 = RL.back_choose_action(state_current1,action1)
					elif action_back == 2:
						action2 = RL.back_choose_action(state_current2,action2)
					elif action_back == 3:
						action3 = RL.back_choose_action(state_current3,action3)
					state, reward, done, x, y, coverage, out, action_back = env.step(action0, action1, action2, action3, x_b, y_b, back = True)
					if out:
						back = True
						i += 1
						j += 1
						if i >= 3:
							a = [0, 1, 2, 3]
							action_back = np.random.choice(a) 
							i = 0
						if j >= 300:
							back = False
							done = True
					else:
						back = False
				# done = True
			if out == False:
				state_new0 = np.stack((state_current0[1], state_current0[2], state_current0[3], state[0]), axis = 0)
				state_new1 = np.stack((state_current1[1], state_current1[2], state_current1[3], state[1]), axis = 0)
				state_new2 = np.stack((state_current2[1], state_current2[2], state_current2[3], state[2]), axis = 0)
				state_new3 = np.stack((state_current3[1], state_current3[2], state_current3[3], state[3]), axis = 0)


				env.store_data(state_current0, action0, reward[0], state_new0)
				env.store_data(state_current1, action1, reward[1], state_new1)
				env.store_data(state_current2, action2, reward[2], state_new2)
				env.store_data(state_current3, action3, reward[3], state_new3)
			# print(state_current0)
			# print(observation_)
				state_current0 = state_new0
				state_current1 = state_new1
				state_current2 = state_new2
				state_current3 = state_new3
			
			# if ((step*4) > 100000) and (step % 2 == 0):
			if out == False:
				if ((step*4) > 100000) and (step % 2 == 0):
					RL.learn()
				coverage = coverage * 100
				
				if coverage >= coverage_max:
					coverage_max = coverage
					cs = format(episode)
				loop += 1
				if loop == 250:
					done = True
			# observation = observation_
			if done:
				if out == False:
					print(loop)
					s, a, r, s_ = env.s1(loop)
					# print(r)
					# print(a)
					pro = (coverage / 100)
					# + (250 / loop)
					# pro = 0.5
					for i in range(loop*4):
						RL.store_transition(s[i], a[i], r[i], s_[i], pro)
					if (loop <= loop_min) and (coverage == 100):
						loop_min = loop
						cs = format(episode)
						path0, path1, path2, path3 = env.get_path() 
					if coverage >= 90:
						print (state[0])
						# cs = format(episode)
						# path0, path1, path2, path3 = env.get_path() 
					if coverage == 100:
						cs = format(episode)
						print(path0)
						print(path1)
						print(path2)
						print(path3)
					
					# print(coverage_max)
					# print(coverage)
					# print(loop_min)
					# print(cs)
				# print(observation)
				loop = 0
				break
			step += 1
	print(loop_min)
	# print(cs)
	if coverage_max == 100:
		print(path0)
		print(path1)
		print(path2)
		print(path3)
	end_time = time.time()
	print('END')
	print(end_time - start_time)
	env.destroy()

def process(o):
	o0 = o
	o0[4][3], o0[5][4], o0[4][5] = 6, 6, 6
	o1 = o
	o1[3][4], o1[5][4], o1[4][5] = 6, 6, 6
	o2 = o
	o2[3][4], o2[4][3], o2[4][5] = 6, 6, 6
	o3 = o
	o3[3][4], o3[4][3], o3[5][4] = 6, 6, 6
	return o0, o1, o2, o3


if __name__ == '__main__':
	env = Maze()
	RL = DeepQNetwork(env.n_actions, env.n_features, 
					)
	env.after(100, run_maze)
	env.mainloop()
	RL.plot_cost()
	RL.plot_cost1()