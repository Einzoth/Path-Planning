import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 9  #地图高度
MAZE_W = 9  #地图宽度
X = UNIT * (MAZE_H // 2)
Y = UNIT * (MAZE_H // 2)
projection_x = 4
projection_y = 4

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = MAZE_H * MAZE_W
        self.title('Grid Map')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))

        self.robot0 = np.zeros((MAZE_W, MAZE_H))
        self.robot0[projection_x-1][projection_y] = 7
        self.robot0[projection_x  ][projection_y]  = -1
        # self.robot0[8][8] = -1

        self.robot1 = np.zeros((MAZE_W, MAZE_H))
        self.robot1[projection_x][projection_y-1] = 7
        self.robot1[projection_x][projection_y  ] = -1
        # self.robot1[8][8] = -1

        self.robot2 = np.zeros((MAZE_W, MAZE_H))
        self.robot2[projection_x+1][projection_y] = 7
        self.robot2[projection_x  ][projection_y] = -1
        # self.robot1[8][8] = -1

        self.robot3 = np.zeros((MAZE_W, MAZE_H))
        self.robot3[projection_x][projection_y+1] = 7
        self.robot3[projection_x][projection_y  ] = -1
        # self.robot1[8][8] = -1

        self.map = np.zeros((MAZE_W, MAZE_H))
        # self.map[projection_x-1][projection_y  ] = 7
        # self.map[projection_x  ][projection_y-1] = 7
        # self.map[projection_x+1][projection_y  ] = 7
        # self.map[projection_x  ][projection_y+1] = 7
        # self.map[projection_x  ][projection_y  ] = -1
        # self.map[8][8] = -1

        self.obs = np.zeros((MAZE_W, MAZE_H))
        self.obs[4][4] = -1
        self.obs[0][8] = -1
        self.obs[7][8] = -1
        self.obs[8][8] = -1
        self.obs[1][4] = -1
        self.obs[8][7] = -1
        self.obs[1][2] = -1
        self.obs[2][2] = -1
        self.obs[2][1] = -1
        self.obs[8][1] = -1
        self.obs[1][8] = -1
        self.obs[6][5] = -1
        # self.obs[1][1] = -1
        self.obs[2][6] = -1

        for i in range (MAZE_W):
            for j in range (MAZE_H):
                if self.obs[i][j] == -1:
                    self.robot0[i][j] = -1
                    self.robot1[i][j] = -1
                    self.robot2[i][j] = -1
                    self.robot3[i][j] = -1
                    # self.map[i][j] = -1

        # self.obs_expansion = np.zeros((MAZE_W+2, MAZE_H+2))
        self.robot_current_location = [[projection_x-1, projection_y   ], 
                                       [projection_x  , projection_y-1 ], 
                                       [projection_x+1, projection_y   ], 
                                       [projection_x  , projection_y+1 ]]

        self.cov = 0
        self.rep = 0
        self.robot0_path = []
        self.robot1_path = []
        self.robot2_path = []
        self.robot3_path = []

        self.map_fin = np.zeros((MAZE_W,MAZE_H))
        self.step_number = 0

        self.x  = [0, 0, 0, 0]
        self.y  = [0, 0, 0, 0]
        self.x_ = [0, 0, 0, 0]
        self.y_ = [0, 0, 0, 0]
        self.reward = 0

        self.data_s  = []
        self.data_r  = []
        self.data_a  = []
        self.data_s_ = []
        self.ab = (MAZE_H * MAZE_W) - 5
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

        obs_origin = np.array([20, 20])
        for i in range (MAZE_W):
            for j in range (MAZE_H):
                if self.obs[i][j] == -1:
                    self.canvas.create_rectangle(
                        obs_origin[0] + j * UNIT -18, obs_origin[1] + i * UNIT - 18,
                        obs_origin[0] + j * UNIT +18, obs_origin[1] + i * UNIT + 18,
                        fill = 'black')

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
        
        # obs_origin = np.array([180, 180])
        # self.obs_rect = self.canvas.create_rectangle(
        #     obs_origin[0] - 18, obs_origin[1] - 18,
        #     obs_origin[0] + 18, obs_origin[1] + 18,
        #     fill='black')
        
        # obs_origin1 = np.array([340, 340])
        # self.obs_rect = self.canvas.create_rectangle(
        #     obs_origin1[0] - 18, obs_origin1[1] - 18,
        #     obs_origin1[0] + 18, obs_origin1[1] + 18,
        #     fill='black')
        self.canvas.pack()
    
    def reset(self):
        self.update()
        time.sleep(0.1)

        self.robot0 = np.zeros((MAZE_W, MAZE_H))
        self.robot0[projection_x-1][projection_y] = 7
        # self.robot0[projection_x  ][projection_y]  = -1
        # self.robot0[8][8] = -1

        self.robot1 = np.zeros((MAZE_W, MAZE_H))
        self.robot1[projection_x][projection_y-1] = 7
        # self.robot1[projection_x][projection_y  ] = -1
        # self.robot1[8][8] = -1

        self.robot2 = np.zeros((MAZE_W, MAZE_H))
        self.robot2[projection_x+1][projection_y] = 7
        # self.robot2[projection_x  ][projection_y] = -1
        # self.robot2[8][8] = -1

        self.robot3 = np.zeros((MAZE_W, MAZE_H))
        self.robot3[projection_x][projection_y+1] = 7
        # self.robot3[projection_x][projection_y  ] = -1
        # self.robot3[8][8] = -1

        self.map = np.zeros((MAZE_W, MAZE_H))

        for i in range (MAZE_W):
            for j in range (MAZE_H):
                if self.obs[i][j] == -1:
                    self.robot0[i][j] = -1
                    self.robot1[i][j] = -1
                    self.robot2[i][j] = -1
                    self.robot3[i][j] = -1
        # self.map[projection_x-1][projection_y  ] = 7
        # self.map[projection_x  ][projection_y-1] = 7
        # self.map[projection_x+1][projection_y  ] = 7
        # self.map[projection_x  ][projection_y+1] = 7
        # self.map[projection_x  ][projection_y  ] = -1
        # self.map[8][8] = -1

        self.robot_current_location = [[projection_x-1, projection_y   ], 
                                       [projection_x  , projection_y-1 ], 
                                       [projection_x+1, projection_y   ], 
                                       [projection_x  , projection_y+1 ]]

        self.cov = 0
        self.rep = 0
        self.robot0_path = []
        self.robot1_path = []
        self.robot2_path = []
        self.robot3_path = []
        self.data_s  = []
        self.data_r  = []
        self.data_a  = []
        self.data_s_ = []

        self.step_number = 0
        self.reward = 0
        self.ab = (MAZE_H * MAZE_W) - 5
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
        
        # obs_origin = np.array([20, 20])
        # for i in range (MAZE_W):
        #     for j in range (MAZE_H):
        #         if self.obs[i][j] == -1:
        #             self.canvas.create_rectangle(
        #                 obs_origin[0] + j * UNIT -18, obs_origin[1] + i * UNIT - 18,
        #                 obs_origin[0] + j * UNIT -18, obs_origin[1] + i * UNIT - 18,
        #                 fill = 'black')
        
        self.canvas.pack()

        # danger = [-7,0,-7,0]
        state = self.robot0 + self.robot1 + self.robot2 + self.robot3

        return state
    
    def step(self, action0, action1, action2, action3, x_b, y_b, back):
        if back:
            for i in range(4):
                self.robot_current_location[i][0] = x_b[i] 
                self.robot_current_location[i][1] = y_b[i]
        
        out = False
        coverage = 0
        done = False
        x  = [0, 0, 0, 0]
        y  = [0, 0, 0, 0]
        x_ = [0, 0, 0, 0]
        y_ = [0, 0, 0, 0]

        for i in range(4):
            x[i]  = self.robot_current_location[i][0]
            y[i]  = self.robot_current_location[i][1]
            x_[i] = self.robot_current_location[i][0]
            y_[i] = self.robot_current_location[i][1]
        self.step_number += 4
        reward = [0, 0, 0, 0]
        G = 0

        state_robot0 = self.canvas.coords(self.rect0)
        base_action0 = np.array([0, 0])
        if action0 == 0:     # up
            x_[0] = x[0] - 1
            if state_robot0[1] > UNIT:
                base_action0[1] -= UNIT
        elif action0 == 1:   # down
            x_[0] = x[0] + 1
            if state_robot0[1] < (MAZE_H - 1) * UNIT:
                base_action0[1] += UNIT
        elif action0 == 2:   # left
            y_[0] = y[0] - 1
            if state_robot0[0] < (MAZE_W - 1) * UNIT:
                base_action0[0] += UNIT
        elif action0 == 3:   # right
            y_[0] = y[0] + 1
            if state_robot0[0] > UNIT:
                base_action0[0] -= UNIT
        
            self.action_back = 0
            out = True

        state_robot1 = self.canvas.coords(self.rect1)
        base_action1 = np.array([0, 0])
        if action1 == 0:     # up
            x_[1] = x[1] - 1
            if state_robot1[1] > UNIT:
                base_action1[1] -= UNIT
        elif action1 == 1:   # down
            x_[1] = x[1] + 1
            if state_robot1[1] < (MAZE_H - 1) * UNIT:
                base_action1[1] += UNIT
        elif action1 == 2:   # left
            y_[1] = y[1] - 1
            if state_robot1[0] < (MAZE_W - 1) * UNIT:
                base_action1[0] += UNIT
        elif action1 == 3:   # right
            y_[1] = y[1] + 1
            if state_robot1[0] > UNIT:
                base_action1[0] -= UNIT

        state_robot2 = self.canvas.coords(self.rect2)
        base_action2 = np.array([0, 0])
        if action2 == 0:     # up
            x_[2] = x[2] - 1
            if state_robot2[1] > UNIT:
                base_action2[1] -= UNIT
        elif action2 == 1:   # down
            x_[2] = x[2] + 1
            if state_robot2[1] < (MAZE_H - 1) * UNIT:
                base_action2[1] += UNIT
        elif action2 == 2:   # left
            y_[2] = y[2] - 1
            if state_robot2[0] < (MAZE_W - 1) * UNIT:
                base_action2[0] += UNIT
        elif action2 == 3:   # right
            y_[2] = y[2] + 1
            if state_robot2[0] > UNIT:
                base_action2[0] -= UNIT

        state_robot3 = self.canvas.coords(self.rect3)
        base_action3 = np.array([0, 0])
        if action3 == 0:     # up
            x_[3] = x[3] - 1
            if state_robot3[1] > UNIT:
                base_action3[1] -= UNIT
        elif action3 == 1:   # down
            x_[3] = x[3] + 1
            if state_robot3[1] < (MAZE_H - 1) * UNIT:
                base_action3[1] += UNIT
        elif action3 == 2:   # left
            y_[3] = y[3] - 1
            if state_robot3[0] < (MAZE_W - 1) * UNIT:
                base_action3[0] += UNIT
        elif action3 == 3:   # right
            y_[3] = y[3] + 1
            if state_robot3[0] > UNIT:
                base_action3[0] -= UNIT
        
        map_test = np.zeros((9,9))

        for i in range(4):
            if (x_[i] == -1) or (x_[i] == MAZE_H) or (y_[i] == -1) or (y_[i] == MAZE_H):
                self.action_back = i
                out  = True
                done = True
                break

        if out == False:
            if self.robot0[x_[0]][y_[0]] == -1:
                self.action_back = 0
                out = True
            elif self.robot1[x_[1]][y_[1]] == -1:
                self.action_back = 1
                out = True
            elif self.robot2[x_[2]][y_[2]] == -1:
                self.action_back = 2
                out =True
            elif self.robot3[x_[3]][y_[3]] == -1:
                self.action_back = 3
                out = True

        if out == False:
            robot00 = self.robot0 * 1
            robot11 = self.robot1 * 1
            robot22 = self.robot2 * 1
            robot33 = self.robot3 * 1
            robot00[x[0]][y[0]]   = 1
            robot00[x_[0]][y_[0]] = 7
            robot11[x[1]][y[1]]   = 1
            robot11[x_[1]][y_[1]] = 7
            robot22[x[2]][y[2]]   = 1
            robot22[x_[2]][y_[2]] = 7
            robot33[x[3]][y[3]]   = 1
            robot33[x_[3]][y_[3]] = 7
            map_y = self.map * 1
            self.map = robot00 + robot11 + robot22 + robot33
            map_test = self.map * 1
            # print(map_y)
            # print(1)
            # print(map_test)

            if map_test[4][4] != -4:
                out = True
                done = True
            else:
                for i in range(9):
                    for j in range(9):
                        if map_test[i][j] > 10:
                            for a in range(4):
                                if (x_[a] == i) and (y_[a] == j):
                                    self.action_back = a
                                    out = True
                                    done = True
                                    break

        ab = 0
        for i in range(9):
            for j in range(9):
                if 4 < self.map[i][j] <= 10:
                    map_test[i][j] = 7
                elif self.map[i][j] >10 or 0 < self.map[i][j] <= 4:
                    map_test[i][j] = 1
                elif self.map[i][j] < 0:
                    map_test[i][j] = -1
                if self.map[i][j] != 0:
                    ab += 1
        # map_test[4][4] = -1

        if out:
            reward = [-1, -1, -1, -1]
            done = False
        else:
            line = []
            for i in range(4):
                line_x = UNIT * (x_[i] + 0.5)
                line_y = UNIT * (y_[i] + 0.5)
                line.append(pow(pow(line_x - X, 2) + pow(line_y - Y, 2), 0.5))
            line_var = np.std(line)
            if line_var >= 20:
                out = True
                done = False
                reward = [-1, -1, -1, -1]
                line_ = [0, 0, 0, 0]
                line_avg = np.mean(line)
                for i in range(4):
                    line_[i] = abs((line[i] - line_avg))
                xx =max(line_)
                for i in range(4):
                    if line_[i] == xx:
                        self.action_back = i

        if out == False:
            self.robot0 = robot00 * 1
            self.robot1 = robot11 * 1
            self.robot2 = robot22 * 1
            self.robot3 = robot33 * 1
            self.canvas.move(self.rect0, base_action0[0], base_action0[1])
            self.canvas.move(self.rect1, base_action1[0], base_action1[1])
            self.canvas.move(self.rect2, base_action2[0], base_action2[1])
            self.canvas.move(self.rect3, base_action3[0], base_action3[1])
            self.robot0_path.append([x_[0], y_[0]])
            self.robot1_path.append([x_[1], y_[1]])
            self.robot2_path.append([x_[2], y_[2]])
            self.robot3_path.append([x_[3], y_[3]])

            ef = (ab - self.cd)
            # reward = ((0.25 * ef) - 0.2)
            for i in range(4):
                if map_y[x_[i]][y_[i]] == 0:
                    reward[i] += 0.1
                else:
                    reward[i] -= 0.05
            self.cd = ab
            self.rep += ef
            coverage = ab/(MAZE_H * MAZE_W)
            # print(coverage)
            # print(1)
            rep = self.rep * 1
            repeat_rate = float((rep/self.step_number))
            # print(repeat_rate)
            # print(2)
            G = (coverage - repeat_rate)
            g = [G, G, G, G]
            # g = [0, 0, 0, 0]
            # print(G)
            if ab == 0:
                reward = [1, 1, 1, 1]
                done = True
            # self.reward += reward + G
            # aa = list(np.add(reward, g))
            self.reward = list(np.add(reward, g))
            # self.reward = list(np.add(self.reward, aa))
            # self.reward = reward + g
            # print(self.reward)
            # self.reward = reward
            # self.reward += G

        for i in range(4):
            self.robot_current_location[i][0] = x_[i]
            self.robot_current_location[i][1] = y_[i]
        self.update()

        state = np.zeros((4, MAZE_W, MAZE_H))
        if out == False:
            for i in range(4):
                state[i] = map_test
                for j in range(4):
                    state[i][x_[j]][y_[j]] = 6
                    if i == j:
                        state[i][x_[j]][y_[j]] = 7
        reward_ = self.reward * 1
        action_back = self.action_back * 1
        # print(state)
        return state, reward_, done, x, y, coverage, out, action_back

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

        return data_robot0, data_robot1, data_robot2, data_robot3
    
    def store_data(self, s, a, r, s_):
        self.data_s.append(s)
        self.data_a.append(a)
        self.data_r.append(r)
        self.data_s_.append(s_)

    def s1(self, loop):
        
        # last_reward = self.data_r[-1]
        # dw = last_reward/(loop*4)
        # # print(dw)
        # i = 0
        # for i in range(loop * 4):
        #     self.data_r[i] = dw * (1 + (i//4))
        #     # self.data_r[i] = dw * i
        s = self.data_s * 1
        a = self.data_a * 1
        r = self.data_r * 1
        s_ = self.data_s_ * 1

        return s, a, r, s_

    def rende(self):
        # time.sleep(0.01)
        self.update()
