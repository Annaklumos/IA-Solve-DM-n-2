from MCTS import MonteCarloTreeSearchNode
import numpy as np
from matplotlib import pyplot

reward_list_cparam = []
si_no= []
for i in range(1,21):
    si_no.append(100*i)
    initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 1, 2, 0, 0, 9],
                              [9, 0, 0, 2, 1, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 9, 9, 9, 9, 9, 9, 9]])
    root = MonteCarloTreeSearchNode(state = initial_state, c_param=1.4, simulation_no=200*i)
    selected_node = root.best_action()
    while not root._results[1]:
        root = MonteCarloTreeSearchNode(state = initial_state, c_param=1.4, simulation_no=200*i)
        selected_node = root.best_action()
    reward_list_cparam.append(root._results[1])
pyplot.plot(si_no, reward_list_cparam)
pyplot.show()

