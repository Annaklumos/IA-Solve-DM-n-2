from MCTS import MonteCarloTreeSearchNode
import numpy as np

def main():
    root = MonteCarloTreeSearchNode(state = initial_state, c_param=5, simulation_no=100)
    selected_node = root.best_action()
    print(root.state)
    print(root.q())
    return selected_node



reward_list_cparam = []
for i in range(1,2):

    initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 1, 2, 0, 0, 9],
                              [9, 0, 0, 2, 1, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 9, 9, 9, 9, 9, 9, 9]])

    root = MonteCarloTreeSearchNode(state = initial_state, c_param=(i/100))
    selected_node = root.best_action()
    reward_list_cparam.append(root._results)
    print("Itération : ", i)
print("Reward list of c_param : ", reward_list_cparam)

reward_list_simu = []
for i in range(1, 2):
    initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 1, 2, 0, 0, 9],
                              [9, 0, 0, 2, 1, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 0, 0, 0, 0, 0, 0, 9],
                              [9, 9, 9, 9, 9, 9, 9, 9]])

    root = MonteCarloTreeSearchNode(state=initial_state, simulation_no=i)
    selected_node = root.best_action()
    reward_list_simu.append(root._results)
    print("Itération : ", i)
print("Reward list of simulation : ", reward_list_simu)


