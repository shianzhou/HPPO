"""RobotRun controller."""
import math

#import gym
from re import X
import time
import numpy as np
import os
import sys
import cv2

import argparse
import platform

from pathlib import Path


from python_scripts.Project_config import path_list, gps_goal

from python_scripts.Webots_interfaces import Darwin
from python_scripts.Project_config import Darwin_config



class RobotRun(Darwin):
    
    # 控制机器人按照action行动的类
    # action:
    def __init__(self, robot, state, action_shouder, action_arm, step, catch_flag, gps1, gps2, gps3, gps4, img_name):
        super().__init__(robot)
        self.img_name = img_name    # 名称
        self.step = step    # 步数
        self.robot_state = state  # 机器人状态
        
        self.gps = [gps1, gps2, gps3, gps4]  # GPS坐标数值列表  
        #GPS设备通常返回一个包含3个值的列表/元组，表示3D空间中的坐标[x, y, z]。
        self.action_shouder = action_shouder  # 动作
        self.action_arm = action_arm  # 动作
        
        #print(f"action_arm: {action_arm}, type: {type(action_arm)}")
        #print(f"action_shouder: {action_shouder}, type: {type(action_shouder)}")
        
        # 计算左臂和左肩的目标位置
        current_left_arm = self.robot_state[5]       # 左臂的当前状态在索引5
        current_left_shoulder = self.robot_state[1]  # 左肩的当前状态在索引1
        left_arm_target = 1.25 * action_arm + 0.25      #通过实际仿真应该改变函数表达式，让他变化更小
        left_shoulder_target = 0.2995 * action_shouder - 0.145     #缩小十倍抓取效果更好

        self.ArmLower = left_arm_target - current_left_arm  # 手臂
        self.ArmLower = max(-0.3, min(0.3, self.ArmLower))  # 限制在-0.3到0.3之间
        self.Shoulder = left_shoulder_target - current_left_shoulder  # 肩部
        self.Shoulder = max(-0.3, min(0.3, self.Shoulder))  # 限制在-0.3到0.3之间

        #print(f"手臂上升: {self.ArmLower}")
        # print(f"肩部上升: {self.Shoulder}")
        # print(f"手臂目标位置: {left_arm_target}")
        # print(f"肩部目标位置: {left_shoulder_target}")

        self.catch_flag = catch_flag  # 抓取标识符
        self.catch_Success_flag = False  # 抓取成功标识符
        #self.small_goal = 0  # 小目标
        # 初始化压力传感器列表
        self.touch = [self.touch_sensors['grasp_L1_1'], 
                      self.touch_sensors['grasp_R1_2']]
        # 压力传感器列表
        self.touch_peng = [self.touch_sensors['arm_L1'], 
                           self.touch_sensors['arm_R1'], 
                           self.touch_sensors['leg_L1'], 
                           self.touch_sensors['leg_L2'], 
                           self.touch_sensors['leg_R1'], 
                           self.touch_sensors['leg_R2']]
        self.future_state = [i for i in self.robot_state]  # 未来状态
        # 下一个状态
        self.next = [self.robot_state[1] + self.Shoulder,  #左肩
                     self.robot_state[0] - self.Shoulder,  #右肩
                     self.robot_state[5] + self.ArmLower,  #左臂
                     self.robot_state[4] - self.ArmLower]  #右臂
        # print(f"左肩: {self.robot_state[1]}")
        # print(f"右肩: {self.robot_state[0]}")
        # print(f"左臂: {self.robot_state[5]}")
        # print(f"右臂: {self.robot_state[4]}")
        self.future_state[1] = self.next[0]  # 未来状态[1] = 下一个状态[0]
        self.future_state[0] = self.next[1]  # 未来状态[0] = 下一个状态[1]
        self.future_state[5] = self.next[2]  # 未来状态[5] = 下一个状态[2]
        self.future_state[4] = self.next[3]  # 未来状态[4] = 下一个状态[3]

    
        self.now_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 当前状态
        self.next_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 下一个状态
        self.touch_value = [0.0, 0.0]  # 压力传感器值
        self.return_flag_list ={'reward':0,
                                'done'  :0,
                                'good'  :0,
                                'goal'  :0,
                                'count' :0} # 标识符列表 reward, done, good, goal, count

    def run(self):

        self.robot.step(32)  # 机器人步长
        acc = self.accelerometer.getValues()  # 加速度传感器值
        gyro = self.gyro.getValues()  # 陀螺仪值
        '''
        x轴：左右方向（正右方）
        y轴：上下方向（正上方）
        z轴：前后方向（正前方）
        '''

        #先不改这一块，或者说留着这个用于判断是否抓取正确台阶的指标

        y1 = Darwin_config.gps_goal[0] - self.gps[1][1]  # 高度 目标位置y1与当前位置y1的差值
        y2 = Darwin_config.gps_goal[0] - self.gps[2][1]  # 高度 目标位置y2与当前位置y2的差值
        z1 = Darwin_config.gps_goal[1] - self.gps[1][2]  # 前进 目标位置z1与当前位置z1的差值
        z2 = Darwin_config.gps_goal[1] - self.gps[2][2]  # 前进 目标位置z2与当前位置z2的差值

        reward1 = 20 - 200 * math.sqrt((y1 * y1) + (z1 * z1))  # 奖励1
        reward2 = 20 - 200 * math.sqrt((y2 * y2) + (z2 * z2))  # 奖励2
        # tan1 = y1 / z1  # 角度1正切值
        # tan2 = y2 / z2  # 角度2正切值
        # angle1 = math.degrees(math.atan(tan1))  # 角度1
        # angle2 = math.degrees(math.atan(tan2))  # 角度2
        # delta_angle1 = abs(angle1 - Darwin_config.standard_angle)   # 角度1与标准角度差值的绝对值
        # delta_angle2 = abs(angle2 - Darwin_config.standard_angle)   # 角度2与标准角度差值的绝对值
        """
        if angle1 < Darwin_config.standard_angle:    # 角度1小于标准角度
            delta_angle1 = Darwin_config.standard_angle - angle1   # 角度1与标准角度的差值
        else:   # 角度1大于标准角度
            delta_angle1 = angle1 - Darwin_config.standard_angle
        if angle2 < Darwin_config.standard_angle:    # 角度2小于标准角度 
            delta_angle2 = Darwin_config.standard_angle - angle2    # 角度2与标准角度的差值
        else:
            delta_angle2 = angle2 - Darwin_config.standard_angle   # 角度2大于标准角度
        # """
        # reward3 = 20 - delta_angle1  # 奖励3=20-角度1与标准角度的差值
        # reward4 = 20 - delta_angle2  # 奖励4=20-角度2与标准角度的差值
        # 添加对夹爪高度的奖励 - 鼓励夹爪位置更高（左边靠前的gps）
        # height_reward = 20 - 200 * math.sqrt((self.gps[0][1] - Darwin_config.min_height) ** 2)
        # # 添加对夹爪前进程度的奖励 - 鼓励夹爪位置更前
        # forward_reward = 20 - 200 * math.sqrt((self.gps[0][2] - Darwin_config.min_forward) ** 2)
        # if reward3 <= -20:  # 奖励3小于-20
        #     reward3 = -20  # 奖励3= -20
        # if reward4 <= -20:  # 奖励4小于-20
        #     reward4 = -20  # 奖励4= -20
        # self.return_flag_list.update({'count':1})
        # 遍历未来状态

        for i in range(len(self.future_state)): 
            if Darwin_config.limit[1][0] <= self.future_state[i] <= Darwin_config.limit[1][1]:    # 角度1在限制范围内
                continue
            else:
                self.return_flag_list.update({'reward':0, 'count':0, 'done':1, 'good':1})
                # 返回下一个状态，奖励，完成，好，目标，计数
                # print('角度1超出限制,catch_flag: 0,done:1--------->341')
                print("设置done=1，原因：future_state超出限制")  
                return self.next_state, \
                       self.return_flag_list['reward'], \
                       self.return_flag_list['done'], \
                       self.return_flag_list['good'], \
                       self.return_flag_list['goal'], \
                       self.return_flag_list['count']
        self.robot.step(32)  # 机器人步长
   
        
        for i in range(3):
            # 加速度传感器值在限制范围内
            if Darwin_config.acc_low[i] < acc[i] < Darwin_config.acc_high[i] and \
               Darwin_config.gyro_low[i] < gyro[i] < Darwin_config.gyro_high[i]:
                continue
            # 加速度传感器值不在限制范围内
            else:
                self.return_flag_list.update({'reward':0, 'count':0, 'done':1, 'good':0})
                # 返回下一个状态，奖励，完成，好，目标，计数
                print('加速度传感器值不在限制范围内,catch_flag: 0,done:1--------->405')
                return self.next_state, \
                       self.return_flag_list['reward'], \
                       self.return_flag_list['done'], \
                       self.return_flag_list['good'], \
                       self.return_flag_list['goal'], \
                       self.return_flag_list['count']
        # 如果catch_flag为0，即还没有抓到
        
        if self.catch_flag == 0.0:
            # 执行动作到下一状态
            self.motors[1].setPosition(self.next[0])  # 电机1设置位置
            self.motors[0].setPosition(self.next[1])  # 电机0设置位置
            self.motors[5].setPosition(self.next[2])  # 电机5设置位置
            self.motors[4].setPosition(self.next[3])  # 电机4设置位置
            # 获取当前4个舵机的位置
            current_positions = [
                self.motors_sensors[1].getValue(),  # 左肩
                self.motors_sensors[0].getValue(),  # 右肩
                self.motors_sensors[5].getValue(),  # 右臂
                self.motors_sensors[4].getValue()   # 左臂
            ]

            # 目标位置
            target_positions = [
                self.next[0],  # 左肩
                self.next[1],  # 右肩
                self.next[2],  # 左臂
                self.next[3],  # 右臂
            ]

            # 循环直到所有舵机都到达目标位置或超时
            timeout = 100  # 设置最大循环次数以防止无限循环
            for _ in range(timeout):
                # 执行一个仿真步长
                if self.robot.step(32) == -1:
                    break

                # 获取当前所有舵机的位置
                current_positions = [self.motors_sensors[i].getValue() for i in range(20)]

                # 检查是否所有舵机都已到达目标位置
                all_in_position = True
                for i in range(len(target_positions)):
                    # 使用一个小的容忍度（例如0.01弧度）来判断是否到达
                    if abs(target_positions[i] - current_positions[i]) > 0.01:
                        all_in_position = False
                        break
                
                # 如果所有舵机都已就位，则退出循环
                if all_in_position:
                    break
                

            self.return_flag_list.update({'done':0, 'reward':reward1 + reward2, 'good':1})
            # 遍历压力传感器
            for m in range(6):
                # 压力传感器值为1.0
                if self.touch_peng[m].getValue() == 1.0:
                    print(f'catch_flag={self.catch_flag}, done=1---------->480ppo')
                 
                    self.return_flag_list.update({'done':1, 'reward':0, 'good':1, 'count':0})
                    # 返回下一个状态，奖励，完成，好，目标，计数
                    return self.next_state, \
                           self.return_flag_list['reward'], \
                           self.return_flag_list['done'], \
                           self.return_flag_list['good'], \
                           self.return_flag_list['goal'], \
                           self.return_flag_list['count']
            # print(f'grasp_L1=', self.touch_sensors['grasp_L1'].getValue())
            # print(f'grasp_L1_1=', self.touch_sensors['grasp_L1_1'].getValue())
            # print(f'grasp_L1_2=', self.touch_sensors['grasp_L1_2'].getValue())
            # print(f'grasp_R1=', self.touch_sensors['grasp_R1'].getValue())
            # print(f'grasp_R1_1=', self.touch_sensors['grasp_R1_1'].getValue())
            # print(f'grasp_R1_2=', self.touch_sensors['grasp_R1_2'].getValue())
            # 遍历压力传感器


            #有一个是1就去尝试抓取，给定一个2000的等待时间，然后去看抓取成功没有
            if self.touch_sensors['grasp_L1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_L1_1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_L1_2'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1_1'].getValue() == 1.0 or \
               self.touch_sensors['grasp_R1_2'].getValue() == 1.0:
                # 打印压力传感器值
                # print("___________")
                # print(self.touch_sensors['grasp_L1'].getValue())
                # print(self.touch_sensors['grasp_L1_1'].getValue())
                # print(self.touch_sensors['grasp_L1_2'].getValue())
                # print(self.touch_sensors['grasp_R1'].getValue())
                # print(self.touch_sensors['grasp_R1_1'].getValue())
                # print(self.touch_sensors['grasp_R1_2'].getValue())

                timer = 0  # 计时器
                self.motors[21].setPosition(-0.5)  # 电机21设置位置 
                self.motors[20].setPosition(-0.5)  # 电机20设置位置
                while self.robot.step(32) != -1:
                    timer += 32  # 计时器增加32 
                    if timer >= 2000:
                        print(timer)
                        print('----------------------------->532')
                        break
                # 遍历压力传感器    
                for j in range(len(self.touch)):
                    self.touch_value[j] = self.touch[j].getValue()  # 压力传感器值
                sucess = np.array_equal(self.touch_value, Darwin_config.touch_T)  # 成功标识符=压力传感器值与目标值相等    
                sucess = np.int(sucess)  # 成功标识符=1
                faild = np.array_equal(self.touch_value, Darwin_config.touch_F)  # 失败标识符=压力传感器值与失败值相等
                faild = np.int(faild)  # 失败标识符=1
                # 失败标识符=1且步长小于等于5
                if faild == 1 and self.step <= 5:
                    self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                    print("失败标识符=1且步长小于等于5")
                    # 写入数据
                    with open(path_list['shu_ju_path_DQN'], 'a') as file:
                        file.write('0')
                        file.write(",")
                        file.close()
                # 失败=1且步长大于5
                elif faild == 1 and self.step > 5:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    print("失败=1且步长大于5")
                # 成功=1
                elif sucess == 1:
                    # 奖励1+奖励2小于20
                    if (reward1 + reward2) < 20:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    # 奖励1+奖励2大于等于20 
                    else:
                        self.return_flag_list.update({'count':0, 'done':1, 'good':1, 'goal':1})
                        print("俺抓到了")  
                        # 写入数据
                        with open(path_list['gps_path_DQN'], 'a') as file:
                            gpss = [self.gps1, self.gps2, self.gps3, self.gps4]  # 目标位置
                            gpss1 = str(gpss)  # 目标位置字符串 
                            #gpss1 = str(self.gps)  # 使用self.gps列表
                            file.write(gpss1)  # 写入目标位置字符串
                            file.write(",")  # 写入逗号
                            file.close()  # 关闭文件
                # 成功=0
                else:
                    # 奖励1+奖励2小于20
                    if (reward1 + reward2) < 20:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                    # 奖励1+奖励2大于等于20 
                    else:
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
            # 遍历电机传感器
            else:
                for i in range(20):
                    # print(f'self.future_state[{i}]={self.future_state[i]}')
                    # print(f'self.motors_sensors[{i}].getValue()={self.motors_sensors[i].getValue()}')
                    # print(f'self.robot_state[{i}]={self.robot_state[i]}')
                    self.next_state[i] = self.motors_sensors[i].getValue()  # 舵机位置传感器值
                    # self.next_state[i] = self.robot_state[i]
                    # print(f'self.next_state[{i}]={self.next_state[i]}')
                    self.cha_zhi = self.next_state[i] - self.future_state[i]  # 当前值与未来值的差值
                    # print(f'i={i}, cha_zhi={self.cha_zhi}, done={self.return_flag_list["done"]}')
                    # if -100 < self.cha_zhi < 100:  # 差值在-0.005到0.005之间
                    if -0.005 < self.cha_zhi < 0.005:  # 差值在-0.005到0.005之间
                        # print(f'i1={i}, 当前值={self.next_state[i]:.4f}, 目标值={self.future_state[i]:.4f}, 差值={self.cha_zhi:.4f}, done={self.return_flag_list["done"]}')
                        # print('----------------------------->577')
                        continue
                    else:
                        print(f'catch_flag={self.catch_flag}, done=1----------------->583PPO')
                        self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                        print(f'i2={i}, 当前值={self.next_state[i]:.4f}, 目标值={self.future_state[i]:.4f}, 差值={self.cha_zhi:.4f}, done={self.return_flag_list["done"]}')
                        break
                        # continue
        # 否则catch_flag为非0，即已经抓到了
        else:
            timer = 0  # 计时器
            self.motors[21].setPosition(-0.5)  # 电机21设置位置
            self.motors[20].setPosition(-0.5)  # 电机20设置位置
            while self.robot.step(32) != -1:
                timer += 32  # 计时器增加32
                if timer >= 2000:  # 计时器大于等于2000
                    break
            # 遍历压力传感器
            for j in range(len(self.touch)):
                self.touch_value[j] = self.touch[j].getValue()  # 压力传感器值
            sucess = np.array_equal(self.touch_value, Darwin_config.touch_T)  # 成功=压力传感器值与目标值相等
            sucess = np.int(sucess)  # 成功=1
            faild = np.array_equal(self.touch_value, Darwin_config.touch_F)  # 失败=压力传感器值与失败值相等
            faild = np.int(faild)  # 失败=1
            # 失败=1且步长小于等于5
            if faild == 1 and self.step <= 5:
                self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                print("失败=1且步长小于等于5")
            # 失败=1且步长大于5
            elif faild == 1 and self.step > 5:
                self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                print("失败=1且步长大于5")
            # 成功=1
            elif sucess == 1:
                # 奖励1+奖励2小于20
                if (reward1 + reward2) < 20:
                    self.return_flag_list.update({'reward':0, 'count':1, 'done':1, 'good':1})
                # 奖励1+奖励2大于等于20
                else:
                    self.return_flag_list.update({'reward':100, 'count':0, 'done':1, 'good':1, 'goal':1})
                    print(self.return_flag_list['reward'])  # 打印奖励
                    print("俺抓到了")  # 打印"俺抓到了"
                    # 写入数据
                    with open(path_list['gps_path_DQN'], 'a') as file:
                        gpss = str([self.gps1, self.gps2, self.gps3, self.gps4])  # 目标位置
                        file.write(gpss)  # 写入目标位置字符串
                        file.write(",")  # 写入逗号
                        file.close()  # 关闭文件
            # 奖励1+奖励2大于等于20
            else:
                # 奖励1+奖励2小于20
                self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                """
                if (reward1 + reward2) < 20:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                else:
                    self.return_flag_list.update({'count':1, 'done':1, 'good':1})
                """
        # print(f'正常情况输出，catch_flag={self.catch_flag}, done=', self.return_flag_list['done'])
        # 返回下一个状态，奖励，完成，好，目标，计数
        return self.next_state, \
               self.return_flag_list['reward'], \
               self.return_flag_list['done'], \
               self.return_flag_list['good'], \
               self.return_flag_list['goal'], \
               self.return_flag_list['count']
    
