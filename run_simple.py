import sys
sys.path.append("./aima-python-uofg_v20192020b")
sys.path.append("/usr/local/lib/python3.7/site-packages")
import gym
import numpy as np
from helpers import *
from uofgsocsai import LochLomondEnv
from search import *
import matplotlib.pyplot as plt

def my_best_first_graph_search_for_vis(problem, f):
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    f = memoize(f, 'f')
    node = Node(problem.initial)

    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = PriorityQueue('min', f)
    frontier.append(node)

    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.pop()
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

def my_astar_search_graph(problem, h=None):
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = my_best_first_graph_search_for_vis(problem,lambda n: n.path_cost + h(n))
    return(iterations, all_node_colors, node)

def search_for_one_solution(problem_id, map_name, plot_or_not):
    problem_id = problem_id
    reward_hole = 0.0
    is_stochastic = False
    if  map_name == '4x4-base':
        n_dim = 4
    else:
        n_dim = 8

    env = LochLomondEnv(problem_id = problem_id, is_stochastic = is_stochastic, reward_hole = reward_hole, map_name_base = map_name)
    env.reset()
    # Create a dict representation of the state space
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    #--------------SOLUTION--------------#
    maze_map = UndirectedGraph(state_space_actions)
    maze_map.locations = state_space_locations
    maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)

    iterations, _, node = my_astar_search_graph(problem=maze_problem, h=None)
    #-------------Trace the solution-----------------#
    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)
    i = 0
    while cnode.state != state_initial_id:
        i += 1
        cnode = cnode.parent
        solution_path.append(cnode)

    solution = []
    solution_x = []
    solution_y = []
    for s in str(solution_path).split('_',-1):
        for s_s in str(s).split('>',-1):
            if s_s.isdigit():
                solution.append(s_s)
    for i in range(int(len(solution)/2)):
        solution_y.append(int(solution[i*2]))
        solution_x.append(int(solution[i*2+1]))

    print("Steps:",i)
    print("Goal state:"+str(solution_path[0]))
    print("Final Solution:",solution_path[::-1])
    print("----------------------------------------")
    env.close()

    plt.cla()
    plt.plot(solution_x[::-1], solution_y[::-1])
    plt.scatter(solution_x[::-1], solution_y[::-1],s=120)
    plt.xlim(0,n_dim-1)
    plt.ylim(n_dim-1,0)
    plt.grid(True)
    plt.title("Simple Agent Solution for Problem%s" % problem_id)
    plt.savefig('./Images/%sx%s maps: Simple Agent Solution for Problem%s.jpg' % (n_dim,n_dim,problem_id))
    print("Figure Saved in Folder 'Images'")
    if plot_or_not == True:
        plt.show()

def simple_eval(map_name):
    plot_or_not = False
    if  map_name == '4x4-base':
        n_dim = 4
    else:
        n_dim = 8
    for i in range(n_dim):
        search_for_one_solution(i, map_name, plot_or_not)

if __name__ == '__main__':
    plot_or_not = True
    problem_id = int(sys.argv[1])
    map_name = str(sys.argv[2])
    search_for_one_solution(problem_id, map_name, plot_or_not)