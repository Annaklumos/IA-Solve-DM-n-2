import numpy as np
from collections import defaultdict

class MonteCarloTreeSearchNode():

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.player = 1
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = self.untried_actions()

    def untried_actions(self):
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        self.player = 3 - self.player
        action = self._untried_actions.pop()
        next_state = self.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=self)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.is_game_over()

    def rollout(self):
        current_rollout_state = self.state

        while not self.is_game_over():

            possible_moves = self.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            self.state = self.move(action)
        return self.game_result()

    def backpropagate(self, result):
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)


    def is_fully_expended(self):
        return len(self._untried_actions)

    def best_child(self, c_param = 0.1):
        choices_weights = []
        for c in self.children:
            choices_weights.append((c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())))
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
        simulation_no = 100

        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)

        return self.best_child(c_param=0.1)

    def get_legal_actions(self):
        player = self.player
        opponent = (3 - player)
        legal_actions = []

        current_state = self.state
        py, px = np.where(current_state == 0)
        for i in range(len(py)):
            if current_state[py[i]][px[i]+1] == opponent:
                x = px[i]+1
                n = 0
                while current_state[py[i]][x] == opponent:
                    x += 1
                    n += 1
                if n > 0 and current_state[py[i]][x] == player:
                    action = (0, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]+1][px[i]+1] == opponent:
                x = px[i]+1
                y = py[i]+1
                n = 0
                while current_state[y][x] == opponent:
                    x += 1
                    y += 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (1, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]+1][px[i]] == opponent:
                y = py[i]+1
                n = 0
                while current_state[y][px[i]] == opponent:
                    y += 1
                    n += 1
                if n > 0 and current_state[y][px[i]] == player:
                    action = (2, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]+1][px[i]-1] == opponent:
                y = py[i]-1
                x = px[i]+1
                n = 0
                while current_state[y][x] == opponent:
                    x -= 1
                    y += 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (3, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]][px[i]-1] == opponent:
                x = px[i]-1
                n = 0
                while current_state[py[i]][x] == opponent:
                    x -= 1
                    n += 1
                if n > 0 and current_state[py[i]][x] == player:
                    action = (4, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]-1][px[i]-1] == opponent:
                x = px[i]-1
                y = py[i]-1
                n = 0
                while current_state[y,x] == opponent:
                    x -= 1
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (5, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]-1][px[i]] == opponent:
                y = py[i]-1
                n = 0
                while current_state[y][px[i]] == opponent:
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][px[i]] == player:
                    action = (6, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]-1][px[i]+1] == opponent:
                x = px[i]+1
                y = py[i]-1
                n = 0
                while current_state[y][x] == opponent:
                    x += 1
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (7, n, px[i], py[i])
                    legal_actions.append(action)

        legal_actions = np.array(legal_actions)
        return legal_actions

    def is_game_over(self):

        empty_case = np.where(self.state == 0)
        if np.size(empty_case) == 0:
            return True
        else:
            legal_action = self.get_legal_actions()
            if np.size(legal_action) == 0:
                return True
            else:
                return False

    def game_result(self):

        player_case = np.count_nonzero(self.state == self.player)
        opponent_case = np.count_nonzero(self.state == (self.player - 2))

        if player_case > opponent_case:
            return 1
        elif player_case < opponent_case:
            return -1
        elif player_case == opponent_case:
            return 0

    def move(self, action):

        coord_x = action[2]
        coord_y = action[3]
        current_state = self.state

        current_state[coord_y][coord_x] = self.player

        self.state = current_state
        return self.state



