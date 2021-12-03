from MCTS import MonteCarloTreeSearchNode
import numpy as np

def main():
    root = MonteCarloTreeSearchNode(state = initial_state)
    selected_node = root.best_action()
    return selected_node

initial_state = np.array([[9, 9, 9, 9, 9, 9, 9, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 1, 2, 0, 0, 9],
                          [9, 0, 0, 2, 1, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 0, 0, 0, 0, 0, 0, 9],
                          [9, 9, 9, 9, 9, 9, 9, 9]])


main()