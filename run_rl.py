import sys
sys.path.append("./aima-python-uofg_v20192020b")
sys.path.append("/usr/local/lib/python3.7/site-packages")
import gym
import numpy as np
from uofgsocsai import LochLomondEnv
import random
import matplotlib.pyplot as plt

def train_for_one_model(problem_id, map_name, train_or_not):
    problem_id = problem_id       # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent
    reward_hole = -0.01      # should be less than or equal to 0.0 (you can fine tune this depending on you RL agent choice)
    is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

    if  map_name == '4x4-base':
        n_dim = 4
        num_episodes = 100000
    else:
        num_episodes = 300000
        n_dim = 8

    env = LochLomondEnv(problem_id = problem_id, is_stochastic = is_stochastic, reward_hole = reward_hole, map_name_base = map_name)

    restart_times = 0
    n_actions_for_episode = 0
    rewards_all_episodes = []
    rewards_all_episodes_per_2000 = []
    x_axis_rewardsvsepisodes = []
    episode_steps = []
    max_steps_per_episode = 10000
    exploration_rate = 0.5
    q_table = np.zeros([env.observation_space.n,env.action_space.n])
    learning_rate = 0.3
    discount = 0.5
    if problem_id == 0 and n_dim == 8:
        learning_rate = 0.2
        discount = 0.8
    if problem_id == 0 and n_dim == 4:
        learning_rate = 0.4
        discount = 0.7
    epsilon_min = 0.005
    epsilon_decay_rate = 0.99995
    shortest_path = 10000
    longest_path = 0
    avg_path = []
    Train_or_not = train_or_not

    if Train_or_not == True:
        #--------------Training Process-----------------#
        for episode in range(num_episodes):
            restart_times += 1
            state = env.reset()
            done = False
            rewards_current_episode = 0
            path = [state]
            if restart_times % 5000 == 0:
                print("\ntraining in progress: #", restart_times)
            for step in range(max_steps_per_episode):
                n_actions_for_episode += 1
                # Exploration - exploitation trade-off
                exploration_exploitation_rate = random.uniform(0, 1)
                epsilon = 0.3

                if exploration_exploitation_rate < epsilon or q_table[state, :].all() == 0:
                    action = env.action_space.sample() # Exploration Method 20% i.e take random action from the available actions
                else:
                    action = np.argmax(q_table[state, :] + np.random.randn(1,4)) # Exploitation Method 80% i.e select the action with max value

                new_state, reward, done, info = env.step(action)
                path.append(new_state)

                # Update Q-table
                q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount * np.max(q_table[new_state, :]) - q_table[state, action])

                state = new_state
                # rewards_current_episode += reward

                if done == True and reward == 1:
                    print("\rEpisode #%s: Finish it within %d steps" % (restart_times, len(path)),end = '')
                    break
                if done == True and reward == -0.01:
                    break

            # epsilon decay
            if epsilon >= epsilon_min:
                epsilon *= epsilon_decay_rate

            # rewards_all_episodes.append(rewards_current_episode)

            # if restart_times % 2000 == 0:
            #     avg_reward_2000 = np.sum(rewards_all_episodes) / (2000 * (restart_times / 2000))
            #     rewards_all_episodes_per_2000.append(avg_reward_2000)
            #     x_axis_rewardsvsepisodes.append(2000 * (restart_times / 2000))
        #---------------SAVE THE MODEL--------------------#
        np.save('%sx%s q_tableP%s.npy' % (n_dim, n_dim, problem_id), q_table)

    #--------FINAL TEST-----------#
    if(train_or_not == True):
        print("\nRunning Test for 50000 times. Please wait...")
    q_table = np.load('%sx%s q_tableP%s.npy' % (n_dim, n_dim, problem_id)) # Load the trained model q table
    env.reset()
    state = env.reset()
    test_total_num = 50000
    test_fail_num = 0
    test_succeed_num = 0
    Avg_rewards_per_1000_episodes = []
    Avg_reward_per_step = []
    Avg_reward_per_episode = []


    for k in range(test_total_num):
        s = env.reset()
        j=0
        rewards_temp = 0

        while j < 1000:

            j += 1
            action = np.argmax(q_table[s,:])
            new_state,r,done,b = env.step(action)
            rewards_temp += r
            s = new_state
            if done and r == -0.01:
                test_fail_num += 1
                break
            if done and r == 1.0:
                avg_path.append(j)
                if shortest_path > j:
                    shortest_path = j
                if longest_path < j:
                    longest_path = j
                test_succeed_num += 1.0
                break
            if j == 1000:
                test_fail_num += 1
        Avg_reward_per_episode.append(rewards_temp)
        Avg_reward_per_step.append(rewards_temp / j)

        if k % 1000 == 0:
            Avg_rewards_per_1000_episodes.append(np.sum(Avg_reward_per_episode) / int(1000 * float(k / 1000)))
            x_axis_rewardsvsepisodes.append(1000 * (k / 1000))
    #--------------OUTPUT FINAL RESULT-----------------#
    if (train_or_not == True):
        print("\n-------------------------------------------")
        print("Average rewards per 1000 episodes:",Avg_rewards_per_1000_episodes[-1])
        print("Average rewards per steps:", Avg_reward_per_step[-1])
        print("Success times:",test_succeed_num)
        print("Failure times:",test_fail_num)
        print("Success rate:",float(test_succeed_num / test_total_num))
        print("Success vs Failure rate:",float(test_succeed_num / test_fail_num))
        print("Steps number (Best case):",shortest_path)
        print("Steps number (Worst case):",longest_path)
        print("Steps number (On average):",np.mean(avg_path))
        print("Learning rate:",learning_rate)

    plt.cla()
    plt.plot(x_axis_rewardsvsepisodes[:], Avg_rewards_per_1000_episodes[:])
    plt.savefig('./Images/%sx%s maps: Average Rewards of Problem%s.jpg' % (n_dim,n_dim,problem_id))
    if (train_or_not == True):
        print("Figure Saved in Folder 'Images'")
        plt.show()
    return test_succeed_num, test_fail_num, shortest_path, longest_path,np.mean(avg_path), learning_rate, Avg_rewards_per_1000_episodes[-1], Avg_reward_per_step[-1]

def rl_eval(map_name):
    train_or_not = False
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
        test_succeed_num, test_fail_num, shortest_path, longest_path, avg_path, learning_rate, Avg_rewards_per_1000_episodes, Avg_reward_per_step = train_for_one_model(i, map_name, train_or_not)
        if i == 0:
            print("Problem ID   |   Successes vs Failures   |   Avgerage rewards per 1000 episodes   |   Average reward per step   |   Shortest path   |   Longest path   |   Average path")
        print("    ",i,"      |         ",round(float(test_succeed_num / test_fail_num),6),"        |                ",round(Avg_rewards_per_1000_episodes,5),"               |          ",round(Avg_reward_per_step,5),"          |       ",shortest_path,  \
              "        |       ",longest_path,"    |      ",round(avg_path,1))
        y.append(float(test_succeed_num / (test_succeed_num + test_fail_num)))

    plt.cla()
    plt.bar(x, y)
    plt.title("Success rate of all maps")
    plt.savefig('./Images/Success rate of all %sx%s problems for RL agent.jpg'%(n_dim,n_dim))
    plt.show()

if __name__ == '__main__':
    train_or_not = True
    problem_id = int(sys.argv[1])
    map_name = str(sys.argv[2])
    train_for_one_model(problem_id, map_name, train_or_not)