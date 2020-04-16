import sys
import os
sys.path.append("/usr/local/lib/python3.7/site-packages")
import numpy as np
from uofgsocsai import LochLomondEnv
import matplotlib.pyplot as plt

def train_for_one_problem(problem_id, map_name):
    problem_id = problem_id        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
    reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this depending on you RL agent choice)
    is_stochastic = False  # should be False for A-star (deterministic search) and True for the RL agent

    env = LochLomondEnv(problem_id = problem_id, is_stochastic = is_stochastic, reward_hole = reward_hole, map_name_base = map_name)
    env.reset()

    done = False
    total_test_num = 60000
    restart_times = 0
    succeed_times = 0
    shortest_path = 100
    one_map_succeed_percentage = []

    for i in range(total_test_num):
        restart_times += 1
        done = False
        n_actions_for_episode = 0
        while not done:
            n_actions_for_episode += 1
            action = env.action_space.sample()  # take random action from the available actions
            observation, reward, done, info = env.step(action)

            if done:
                print("\rProblem:%s Episodes #%s / 60000" % (problem_id,restart_times),end='')
                if reward == 1.0:
                    if shortest_path > n_actions_for_episode:
                        shortest_path = n_actions_for_episode
                    succeed_times += 1
                else:
                    env.reset()

    print("\nSucceed Times:",succeed_times)
    print("Total Times:",total_test_num)
    print("Shortest path:",shortest_path)

    one_map_succeed_percentage = float(succeed_times / 60000)
    return one_map_succeed_percentage
    env.close()

def random_eval(map_name):
    map_4x4_names = ['Map0','Map1','Map2','Map3']
    map_8x8_names = ['Map0','Map1','Map2','Map3','Map4','Map5','Map6','Map7']
    y = []
    if  map_name == '4x4-base':
        n_dim = 4
        x = map_4x4_names
    else:
        n_dim = 8
        x = map_8x8_names
    for i in range(n_dim):
        y.append(train_for_one_problem(i, map_name))

    plt.cla()
    plt.bar(x, y)
    plt.title('Success Rate')
    plt.savefig('./Images/%sx%s maps: Random Agent Performance.jpg' % (n_dim, n_dim))
    print("Figure Saved in Folder 'Images'")
    # plt.show()

if __name__ == '__main__':
    problem_id = int(sys.argv[1])
    map_name = str(sys.argv[2])
    train_for_one_problem(problem_id, map_name)



