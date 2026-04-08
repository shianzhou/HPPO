import sys
import os
# 添加项目路径到系统路径
sys.path.append('E:\\project_MultiAgent_h')
# from python_scripts.DQN import DQN_episoid
from python_scripts.PPO import PPO_episoid_1
# from python_scripts.SAC import SAC_episoid
from python_scripts.Project_config import path_list

def main():
    # 直接指定模型路径
    #model_path = "D:/project_MultiAgent/python_scripts/DQN/checkpoint/dqn_model_0.ckpt"

    #print("将使用DQN进行训练")
    #DQN_episoid.DQN_episoid()#model_path=model_path

    print("将使用PPO进行训练")
    PPO_episoid_1.PPO_episoid_1()

    #print("将使用SAC进行训练")
    #SAC_episoid.SAC_episoid()

if __name__ == '__main__':
    print("_________")
    with open(path_list['resetFlag'], 'r+') as file:
        file.write('1')
    main()


# import sys

# # 添加项目路径到系统路径
# sys.path.append('E:\\project_MultiAgent_h')
# # sys.path.append('C:\\Users\\lenovo\\AppData\\Local\\Programs\\Webots\\projects\\robots\\robotis\\darwin-op\\libraries\\managers')
# from python_scripts.PPO import PPO_episoid_1
# from python_scripts.Project_config import path_list
# from controller import Robot


# def main():
#     robot = Robot()
#     # =========================
#     # 默认：训练模式（当前启用）
#     # =========================
#     # print('将使用PPO进行训练')
#     # PPO_episoid_1.PPO_episoid_1(robot)

#     # =========================
#     # 测试模式（需要时取消注释）
#     # 使用说明：
#     # 1) 注释掉上面的两行训练代码
#     # 2) 取消注释下面测试代码
#     # =========================
#     from python_scripts.PPO.PPO_climb_test import run_climb_test
#     print('将使用PPO进行测试（爬梯）')
#     run_climb_test(
#         robot=robot,
#         catch_model_path='E:/project_MultiAgent_h/python_scripts/PPO/checkpoint/catch',
#         tai_model_path='E:/project_MultiAgent_h/python_scripts/PPO/checkpoint/tai',
#         max_cycles=20,
#         result_file='E:/project_MultiAgent_h/python_scripts/PPO/log/tai_log/climb_test_result.json'
#     )


# if __name__ == '__main__':
#     print("_________")
#     with open(path_list['resetFlag'], 'r+') as file:
#         file.write('1')
#     main()
