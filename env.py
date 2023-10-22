import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 9  #地图高度
MAZE_W = 9  #地图宽度
x = UNIT * (MAZE_H // 2)
y = UNIT * (MAZE_H // 2)

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = MAZE_H * MAZE_W
        self.title('Grid Map')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))

        self.robot0 = np.zeros((MAZE_W, MAZE_H))
        self.robot0[3][4] = 7
        self.robot0[4][4] = -1

        self.robot1 = np.zeros((MAZE_W, MAZE_H))
        self.robot1[4][3] = 7
        self.robot1[4][4] = -1

        self.robot2 = np.zeros((MAZE_W, MAZE_H))
        self.robot2[5][4] = 7
        self.robot2[4][4] = -1

        self.robot3 = np.zeros((MAZE_W, MAZE_H))
        self.robot3[4][5] = 7
        self.robot3[4][4] = -1

        self.map = np.zeros((MAZE_W, MAZE_H))
        self.map[3][4] = 7
        self.map[4][3] = 7
        self.map[5][4] = 7
        self.map[4][5] = 7
        self.map[4][4] = -1

        self.obs_expansion = np.zeros((MAZE_W+2, MAZE_H+2))


        self.robot0_current_location = [3,4]
        self.robot1_current_location = [4,3]
        self.robot2_current_location = [5,4]
        self.robot3_current_location = [4,5]
        self.cov = 0
        self.rep = 0
        # self.action_list = []
        self.robot0_path = []
        self.robot1_path = []
        self.robot2_path = []
        self.robot3_path = []
        # self.obs_path = []
        # self.robot_path_fin = []

        # self.obs_path_fin = []
        self.map_fin = np.zeros((MAZE_W,MAZE_H))
        self.step_number = 0
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.x3 = 0
        self.y3 = 0
        self.x = [0, 0, 0, 0]
        self.y = [0, 0, 0, 0]
        self.reward = 0
        # self.al = [2,2,2,1,1,1,1,1,3,3,3,2,2,2,0,0,0,0,0,3,3,3]
        # self.all = 0
        self.data_s  = []
        self.data_r  = []
        self.data_a  = []
        self.data_s_ = []
        self.cd = 0
        self.rep = 0
        self.action_back = -1


        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)
        
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin0 = np.array([140, 180])
        self.rect0 = self.canvas.create_rectangle(
            origin0[0] - 18, origin0[1] - 18,
            origin0[0] + 18, origin0[1] + 18,
            fill='red')
        
        origin1 = np.array([180, 140])
        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 18, origin1[1] - 18,
            origin1[0] + 18, origin1[1] + 18,
            fill='red')
        
        origin2 = np.array([220, 180])
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 18, origin2[1] - 18,
            origin2[0] + 18, origin2[1] + 18,
            fill='red')
        
        origin3 = np.array([180, 220])
        self.rect3 = self.canvas.create_rectangle(
            origin3[0] - 18, origin3[1] - 18,
            origin3[0] + 18, origin3[1] + 18,
            fill='red')
        
        obs_origin = np.array([180, 180])
        self.obs_rect = self.canvas.create_rectangle(
            obs_origin[0] - 18, obs_origin[1] - 18,
            obs_origin[0] + 18, obs_origin[1] + 18,
            fill='black')
        
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(0.1)

        self.robot0 = np.zeros((MAZE_W, MAZE_H))
        self.robot0[3][4] = 7
        self.robot0[4][4] = -1

        self.robot1 = np.zeros((MAZE_W, MAZE_H))
        self.robot1[4][3] = 7
        self.robot1[4][4] = -1

        self.robot2 = np.zeros((MAZE_W, MAZE_H))
        self.robot2[5][4] = 7
        self.robot2[4][4] = -1

        self.robot3 = np.zeros((MAZE_W, MAZE_H))
        self.robot3[4][5] = 7
        self.robot3[4][4] = -1

        self.map = np.zeros((MAZE_W, MAZE_H))
        self.map[3][4] = 7
        self.map[4][3] = 7
        self.map[5][4] = 7
        self.map[4][5] = 7
        self.map[4][4] = -1
        # self.al = [2,2,2,1,1,1,1,1,3,3,3,2,2,2,0,0,0,0,0,3,3,3]
        # self.all = 0
        self.cov = 0
        self.rep = 0
        self.robot0_path = []
        self.robot1_path = []
        self.robot2_path = []
        self.robot3_path = []
        # self.obs_path = []
        # self.path[0] = [0,0]
        self.step_number = 0
        self.reward = 0
        self.cd = 0
        self.rep = 0
        self.action_back = -1

        self.canvas.delete(self.rect0)
        self.canvas.delete(self.rect1)
        self.canvas.delete(self.rect2)
        self.canvas.delete(self.rect3)

        origin0 = np.array([140, 180])
        self.rect0 = self.canvas.create_rectangle(
            origin0[0] - 18, origin0[1] - 18,
            origin0[0] + 18, origin0[1] + 18,
            fill='red')
        
        origin1 = np.array([180, 140])
        self.rect1 = self.canvas.create_rectangle(
            origin1[0] - 18, origin1[1] - 18,
            origin1[0] + 18, origin1[1] + 18,
            fill='red')
        
        origin2 = np.array([220, 180])
        self.rect2 = self.canvas.create_rectangle(
            origin2[0] - 18, origin2[1] - 18,
            origin2[0] + 18, origin2[1] + 18,
            fill='red')
        
        origin3 = np.array([180, 220])
        self.rect3 = self.canvas.create_rectangle(
            origin3[0] - 18, origin3[1] - 18,
            origin3[0] + 18, origin3[1] + 18,
            fill='red')
        # self.canvas.delete(self.obs_rect)
        # obs_origin = np.array([260, 60])
        # self.obs_rect = self.canvas.create_rectangle(
        #     obs_origin[0] - 18, obs_origin[1] - 18,
        #     obs_origin[0] + 18, obs_origin[1] + 18,
        #     fill='black')
        self.canvas.pack()

        # danger = [-7,0,-7,0]
        state = self.robot0 + self.robot1 + self.robot2 + self.robot3

        return state
    
    def step(self, action0, action1, action2, action3, x0b, y0b, x1b, y1b, x2b, y2b, x3b, y3b, back):
        if back:
            self.robot0_current_location[0] = x0b
            self.robot0_current_location[1] = y0b
            self.robot1_current_location[0] = x1b
            self.robot1_current_location[1] = y1b
            self.robot2_current_location[0] = x2b
            self.robot2_current_location[1] = y2b
            self.robot3_current_location[0] = x3b
            self.robot3_current_location[1] = y3b
            # self.robot0_path.pop()
            # self.robot1_path.pop()
            # self.robot2_path.pop()
            # self.robot3_path.pop()
            # self.obs_path.pop()
            # self.action_list.pop()
            # self.all -= 1
        
        out = False
        coverage = 0
        done = False
        x0 = self.robot0_current_location[0]
        y0 = self.robot0_current_location[1]
        x1 = self.robot1_current_location[0]
        y1 = self.robot1_current_location[1]
        x2 = self.robot2_current_location[0]
        y2 = self.robot2_current_location[1]
        x3 = self.robot3_current_location[0]
        y3 = self.robot3_current_location[1]
        x0_ = self.robot0_current_location[0]
        y0_ = self.robot0_current_location[1]
        x1_ = self.robot1_current_location[0]
        y1_ = self.robot1_current_location[1]
        x2_ = self.robot2_current_location[0]
        y2_ = self.robot2_current_location[1]
        x3_ = self.robot3_current_location[0]
        y3_ = self.robot3_current_location[1]
        # self.robot_path[self.step_number] = self.robot_current_location
        self.step_number += 1
        reward = 0
        G = 0

        state_robot0 = self.canvas.coords(self.rect0)
        base_action0 = np.array([0, 0])
        if action0 == 0:     # up
            x0_ = x0 - 1
            if state_robot0[1] > UNIT:
                base_action0[1] -= UNIT
        elif action0 == 1:   # down
            x0_ = x0 + 1
            if state_robot0[1] < (MAZE_H - 1) * UNIT:
                base_action0[1] += UNIT
        elif action0 == 2:   # left
            y0_ = y0 - 1
            if state_robot0[0] < (MAZE_W - 1) * UNIT:
                base_action0[0] += UNIT
        elif action0 == 3:   # right
            y0_ = y0 + 1
            if state_robot0[0] > UNIT:
                base_action0[0] -= UNIT
        if robot0[x0_][y0_] == -1:
            self.action_back == 0

        state_robot1 = self.canvas.coords(self.rect1)
        base_action1 = np.array([0, 0])
        if action1 == 0:     # up
            x1_ = x1 - 1
            if state_robot1[1] > UNIT:
                base_action1[1] -= UNIT
        elif action1 == 1:   # down
            x1_ = x1 + 1
            if state_robot1[1] < (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
        elif action1 == 2:   # left
            y1_ = y1 - 1
            if state_robot1[0] < (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
        elif action1 == 3:   # right
            y1_ = y1 + 1
            if state_robot1[0] > UNIT:
                base_action1[0] -= UNIT
        if self.robot1[x1_][y1_] == -1:
            self.action_back = 1

        state_robot2 = self.canvas.coords(self.rect2)
        base_action2 = np.array([0, 0])
        if action2 == 0:     # up
            x2_ = x2 - 1
            if state_robot2[1] > UNIT:
                base_action2[1] -= UNIT
        elif action2 == 1:   # down
            x2_ = x2 + 1
            if state_robot2[1] < (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action2 == 2:   # left
            y2_ = y2 - 1
            if state_robot2[0] < (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action2 == 3:   # right
            y2_ = y2 + 1
            if state_robot2[0] > UNIT:
                base_action2[0] -= UNIT
        if self.robot2[x2_][y2_] == -1:
            self.action_back = 2

        state_robot3 = self.canvas.coords(self.rect3)
        base_action3 = np.array([0, 0])
        if action3 == 0:     # up
            x3_ = x3 - 1
            if state_robot3[1] > UNIT:
                base_action3[1] -= UNIT
        elif action3 == 1:   # down
            x3_ = x3 + 1
            if state_robot3[1] < (MAZE_H - 1) * UNIT:
                base_action3[1] += UNIT
        elif action3 == 2:   # left
            y3_ = y3 - 1
            if state_robot3[0] < (MAZE_W - 1) * UNIT:
                base_action3[0] += UNIT
        elif action3 == 3:   # right
            y3_ = y3 + 1
            if state_robot3[0] > UNIT:
                base_action3[0] -= UNIT
        if self.robot3[x3_][y3_] == -1:
            self.action_back = 3
        # self.canvas.move(self.rect, base_action[0], base_action[1])
        # self.robot[x][y] = 1
        # self.robot[x_][y_] = 7
        # cheak = True
        # while cheak:


            # if (obs_x_ == -1) or (obs_y_ == -1) or (obs_x_ == 9) or (obs_y_ == 9):
            #     obs_x_ = obs_x
            #     obs_y_ = obs_y
            #     cheak = True
            # else:
            #     cheak = False

        # self.canvas.move(self.obs_rect, obs_base_action[0], obs_base_action[1])
        obs_negative = -1 * self.obs
        obs_positive = 1 * self.obs
        self.obs = obs_positive + obs_negative

        #奖励函数
        if (x0_ == -1) or (y0_ == -1) or (x0_ == 9) or (y0_ == 9) or (x1_ == -1) or (y1_ == -1) or (x1_ == 9) or (y1_ == 9) or(x2_ == -1) or (y2_ == -1) or (x2_ == 9) or (y2_ == 9) or (x3_ == -1) or (y3_ == -1) or (x3_ == 9) or (y3_ == 9):
            # reward = -1
            out = True
            # done = True
        else:
            # coordinate0 = self.robot0[x0_][y0_]
            self.robot0[x0][y0] = 1
            self.robot0[x0_][y0_] = 7
            # coordinate1 = self.robot1[x1_][y1_]
            self.robot1[x1][y1] = 1
            self.robot1[x1_][y1_] = 7
            # coordinate2 = self.robot2[x2_][y2_]
            self.robot2[x2][y2] = 1
            self.robot2[x2_][y2_] = 7
            # coordinate3 = self.robot3[x3_][y3_]
            self.robot3[x3][y3] = 1
            self.robot3[x3_][y3_] = 7
            robot0 = self.robot0 * 1
            robot1 = self.robot1 * 1
            robot2 = self.robot2 * 1
            robot3 = self.robot3 * 1
            obs = self.obs * 1
            self.obs_expansion = np.pad(obs, pad_width = 1,mode = 'constant', constant_values = -1)
            self.map = obs + robot0 + robot1 + robot2 + robot3
            map_test =  1 * self.map
            map_expansion = 1 * self.map
            map_expansion = np.pad(map_expansion, pad_width = 1,mode = 'constant', constant_values = -1)
            if map_test[4][4] != -4:
                # reward = -1
                out = True
            else:
                map_test[4][4] = -1
                ab = 0
                for i in range(9):
                    for j in range(9):
                        if map_test[i][j] > 10:
                            map_test[i][j] = -5
                            # reward = -1
                            out = True

                        else:
                            if map_test[i][j] == 0:
                                ab += 1
                            elif 1< map_test[i][j] <=4:
                                map_test[i][j] = 1
                            elif 4< map_test[i][j] <= 10:
                                map_test[i][j] = 7
                if out == False:
                    line = []
                    x_0 = UNIT * (x0_ + 0.5)
                    y_0 = UNIT * (y0_ + 0.5)
                    line.append(pow(pow(x_0-x, 2)+pow(y_0-y, 2), 0.5))
                    x_1 = UNIT * (x1_ + 0.5)
                    y_1 = UNIT * (y1_ + 0.5)
                    line.append(pow(pow(x_1-x, 2)+pow(y_1-y, 2), 0.5))
                    x_2 = UNIT * (x2_ + 0.5)
                    y_2 = UNIT * (y2_ + 0.5)
                    line.append(pow(pow(x_2-x, 2)+pow(y_2-y, 2), 0.5))
                    x_3 = UNIT * (x3_ + 0.5)
                    y_3 = UNIT * (y3_ + 0.5)
                    line.append(pow(pow(x_3-x, 2)+pow(y_3-y, 2), 0.5))
                    line_var = np.std(line)
                    if line_var >= 10:
                        out = True

        if out:
            reward = -1
        else:
            self.canvas.move(self.rect0, base_action0[0], base_action0[1])
            self.canvas.move(self.rect1, base_action1[0], base_action1[1])
            self.canvas.move(self.rect2, base_action2[0], base_action2[1])
            self.canvas.move(self.rect3, base_action3[0], base_action3[1])
            self.robot0_path.append([x0_,y0_])
            self.robot1_path.append([x1_,y1_])
            self.robot2_path.append([x2_,y2_])
            self.robot3_path.append([x3_,y3_])
            if ab == 0:
                reward = 5
                done = True
            ef = 4 - (self.cd - ab)
            self.cd = ab
            self.rep += ef
            coverage = ((MAZE_H * MAZE_W) - ab)/(MAZE_H * MAZE_W)
            rep = self.rep * 1
            repeat_rate = float((rep/self.step_number))
            reward = 0.25 * (4 - ef)
            G = (coverage - repeat_rate)
        self.reward += reward + G

            








            # elif coordinate == 1:
            #     self.rep += 1
            #     if (map_expansion[x][y+1] != 0) and (map_expansion[x+2][y+1] != 0) and (map_expansion[x+1][y+2] != 0) and (map_expansion[x+1][y] != 0):
            #         reward = 0
            #         done = False
            #         out = False
            #     else:
            #         reward = -0.1
            #         done = False
            #         out = False


        # cover = self.cov * 1
        # if  cover == 80:
        #     reward = 5
        #     done = True

        # if out == False:
        #     self.canvas.move(self.rect, base_action[0], base_action[1])
        #     self.canvas.move(self.obs_rect, obs_base_action[0], obs_base_action[1])
        # repeat = self.rep * 1
        # repeat_rate = float((repeat/self.step_number))
        # coverage = float((cover/80))
        # G = (coverage - repeat_rate)
        # self.reward += reward + G
        # self.robot0_path.append([x0_,y0_])
        # self.robot1_path.append([x1_,y1_])
        # self.robot2_path.append([x2_,y2_])
        # self.robot3_path.append([x3_,y3_])
        
        # self.obs_current_location = [obs_x_, obs_y_]
        # self.robot_path.append(self.robot_current_location)
        # self.obs_path.append(self.obs_current_location)
        # self.action_list.append(action)
        # print(self.robot_path)
        self.x0 = x0_
        self.y0 = y0_
        self.x1 = x1_
        self.y1 = y1_
        self.x2 = x2_
        self.y2 = y2_
        self.x3 = x3_
        self.y3 = y3_
        self.update()

        # self.robot_path_fin = self.robot_path * 1
        # self.obs_path_fin = self.obs_path * 1

        # for i in range(0,MAZE_H,1):
        #     for j in range(0,MAZE_H,1):
        #         if self.map[i][j] < 0:
        #             self.map[i][j] = -1

        state0 = map_test
        state0[x1_][y1_], state0[x2_][y2_], state0[x3_][y3_] = 6, 6, 6
        state1 = map_test
        state1[x0_][y0_], state1[x2_][y2_], state1[x3_][y3_] = 6, 6, 6
        state2 = map_test
        state2[x0_][y0_], state2[x1_][y1_], state2[x3_][y3_] = 6, 6, 6
        state3 = map_test
        state3[x0_][y0_], state3[x1_][y1_], state3[x2_][y2_] = 6, 6, 6


        reward_ = self.reward * 1
        # print(state_)
        return state0, state1, state2, state3, reward_, done, x0, y0, x1, y1, x2, y2, x3, y3, coverage, out

    # def render(self):
    #     self.update()
    #     #优化点2
    #     # x = self.x * 1
    #     # y = self.y * 1
    #     obs = self.obs_expansion * 1
    #     # danger = [0,0,0,0]
    #     # if (-1 < x < 9) and (-1 < y < 9):
    #     #     u1 = (obs[x][y+1]) * 7
    #     #     d1 = (obs[x+2][y+1]) * 7
    #     #     l1 = (obs[x+1][y]) * 7
    #     #     r1 = (obs[x+1][y+2]) * 7
    #     #     danger = [u1,d1,l1,r1]
    #     return danger
    
    def get_path(self):
        # robot_path = []
        # obs_path = []
        
        data_robot0 = self.robot0_path * 1
        data_robot1 = self.robot1_path * 1
        data_robot2 = self.robot2_path * 1
        data_robot3 = self.robot3_path * 1
        # i = 0
        # for i in range (len(data_robot) - 1):
        #     j = i + 1
        #     if (data_robot[i] == data_robot[j]).all():
        #         robot_path.append(data_robot[i])
        #         obs_path.append(data_obs[i])
            
        return data_robot0, data_robot1, data_robot2
    
    def store_data(self, s, a, r, s_):
        self.data_s.append(s)
        self.data_a.append(a)
        self.data_r.append(r)
        self.data_s_.append(s_)

    def s1(self, loop):
        
        last_reward = self.data_r[-1]
        dw = last_reward/loop
        i = 0
        for i in range(loop):
            self.data_r[i] = dw * (1 + (i//4))
        s = self.data_s * 1
        a = self.data_a * 1
        r = self.data_r * 1
        s_ = self.data_s_ * 1

        return s, a, r, s_

    # def get_action(self):
    #     action_data = self.action_list * 1
    #     return action_data
 

    def rende(self):
        # time.sleep(0.01)
        self.update()
