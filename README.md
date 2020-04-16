# AI-Agents-for-Maze-Problem
The task contains three different agents. A random agent, a simple agent and a RL agent. The problem is similar to the Frozen Lake problem which introduced by Open AI Gym. It requires the agent to “learn” how to get across the lake from the start point to the goal point and not to fall into a hole.

./aima-python-uofg_v20192020b is contained as a zip file in the file list


The task contains three different agents. A random agent, a simple agent and a RL agent. The problem is similar to the Frozen Lake problem which introduced by Open AI Gym. It requires the agent to “learn” how to get across the lake from the start point to the goal point and not to fall into a hole. The environment setting for each agent is different. According to the PEAS analysis method, for Environment, three agents face the same environment introduces above. For Actuators and Sensors, respectively, first random agent doesn’t have any prior knowledge about the lake map or the reward of each step. It just tries all actions randomly to finish the task. Second simple agent uses A* method to decide which action to take in the next step. It is allowed to read the location of all objects on the maze map, and it takes noise free step. Besides this agent can fully understand the reward of each step and know what goal it needs to achieve. The third RL agent uses Q- learning method to get across the lake. It gets perfect information about the current state and thus available actions in that state, but it has no prior knowledge about the state- space in general. Because this agent can learn from observation and actions, so the rewards of each step is not introduced in advance as well. One important thing is the step which this agent would like to take is noise contained, meaning it may accidently move to a location which it does not want to reach. As for Performance, after training process, I will run a test to find how many times this agent is going to make it out of all times of attempts.

The develop environment is Python3.7 and I am mainly using PyCharm to finish all the codes. I use some external packages like gym, numpy, random and matplotlib. Besides I also use existing codes from the AIMA toolbox, uofgsocsai.py and helpers.py.



>for myself reference: Report is saved in */Users/ericlee/Documents/PythonProjects/Agents_for_Frozen_Lake_Problem/AI Report*

















