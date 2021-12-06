import numpy as np
from collections import defaultdict
from etat_jeu import State


class MonteCarloTreeSearchNode:

    def __init__(self, state, c_param=0.1, simulation_no=100, parent=None, parent_action=None):
        self.state = State(first_state=state)  # état du plateau
        self.c_param = c_param
        self.simulation_no = simulation_no
        self.player = 2  # joueur dont c'est le tour
        self.parent = parent  # noeud précédent, None pour r
        self.parent_action = parent_action  # action précédente
        self.children = []  # liste des descendants
        self._number_of_visits = 0  # nombre de visites du noeud
        self._results = defaultdict(int)  # dictionnaire
        self._results[1] = 0  # nombre de wins
        self._results[-1] = 0  # nombre de looses
        self._untried_actions = self.untried_actions()

    def untried_actions(self):  # liste des actions à tester
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):  # renvoie la différence win-loose
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):  # nombre de fois qu'un noeud est visité
        return self._number_of_visits

    def expand(self):  # choisit une action parmi les possibles et renvoie le noeud créé
        action = self._untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=action, c_param=self.c_param, simulation_no=self.simulation_no)

        self.children.append(child_node)

        return child_node

    def is_terminal_node(self):  # teste si le noeud est une feuille
        return self.state.is_game_over()

    def rollout(self):  # continue de dérouler la partie
        current_rollout_state = State(first_state=self.state)

        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expended(self):  # teste si toutes les actions ont été essayées
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):  # upper confidence bound

        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expended():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = self.simulation_no

        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            print(self.state.state)

        return self.best_child(self.c_param)

