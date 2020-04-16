from run_random import *
from run_simple import *
from run_rl import *

if __name__ == '__main__':
    map_name = str(sys.argv[1])
    print("Running Random Agent Test...")
    random_eval(map_name)
    print("Running Simple Agent Test...")
    simple_eval(map_name)
    print("Running RL Agent Test...")
    rl_eval(map_name)