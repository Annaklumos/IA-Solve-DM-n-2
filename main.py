from MCTS import MonteCarloTreeSearchNode
import numpy as np

def main():
    root = MonteCarloTreeSearchNode(state = initial_state, c_param=5, simulation_no=200)
    selected_node = root.best_action()
    print(root.state)
    print(root.q())
    return selected_node

initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 1, 2, 0, 0, 9],
                          [9, 0, 0, 2, 1, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 9, 9, 9, 9, 9, 9, 9]])

c_param_list = []
for i in range(1,101):
    root = MonteCarloTreeSearchNode(state=initial_state, c_param=1/101)
    selected_node = root.best_action()
    c_param_list.append(root.q())
print(c_param_list)
