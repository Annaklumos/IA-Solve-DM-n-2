from MCTS import MonteCarloTreeSearchNode
import numpy as np


reward_list_cparam = []
cparam = []
for i in range(100):
    c_param.append(i/10)
    initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 1, 2, 0, 0, 9],
                              [9, 0, 0, 2, 1, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 9, 9, 9, 9, 9, 9, 9]])
    root = MonteCarloTreeSearchNode(state = initial_state, c_param=i/10, simulation_no=100)
    selected_node = root.best_action()
    reward_list_cparam.append(root._results[1])
print(reward_list_cparam)



