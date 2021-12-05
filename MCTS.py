import numpy as np
from collections import defaultdict


class MonteCarloTreeSearchNode():

    def __init__(self, state, parent=None, parent_action=None):
        self.state = state #état du plateau
        self.player = 2 #joueur dont c'est le tour
        self.parent = parent #noeud précédent, None pour r
        self.parent_action = parent_action #action précédente
        self.children = [] #liste des descendants 
        self._number_of_visits = 0# nombre de visites du noeud 
        self._results = defaultdict(int) #dictionnaire
        self._results[1] = 0  #nombre de wins
        self._results[-1] = 0 #nombre de looses
        self._untried_actions = self.untried_actions()

    def untried_actions(self): #liste des actions à tester
        self._untried_actions = self.get_legal_actions()
        return self._untried_actions

    def q(self): #renvoie la différence win-loose
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses 

    def n(self): #nombre de fois qu'un noeud est visité
        return self._number_of_visits

    def expand(self): # choisit une action parmi les possibles et renvoie le noeud créé
        self.player = 3 - self.player
        action = self._untried_actions.pop()
        next_state = self.move(action)
        child_node = MonteCarloTreeSearchNode(
            next_state, parent=self, parent_action=self)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self): #teste si le noeud est une feuille
        return self.is_game_over()

    def rollout(self): #continue de dérouler la partie
        while not self.is_game_over():
            possible_moves = self.get_legal_actions()

            action = self.rollout_policy(possible_moves)
            self.state = self.move(action)
            self.player = 3 - self.player # modif ici!
        return self.game_result()

    def backpropagate(self, result): 
        self._number_of_visits += 1
        self._results[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expended(self): #teste si toutes les actions ont été essayées
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1): #upper confidence bound

        choices_weights = [( c.q() / c.n() ) + c_param * np.sqrt(( 2 * np.log(self.n()) / c.n() )) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):

        current_node = self

        while not current_node.is_terminal_node():
            if not current_node.is_fully_expended():
                print("Ok")
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self):
        simulation_no = 5

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
        if len(py) == 32:
            player = 1
            opponent = 2
        for i in range(len(py)):
            if current_state[py[i]][px[i] + 1] == opponent:
                x = px[i] + 1
                n = 0
                while current_state[py[i]][x] == opponent:
                    x += 1
                    n += 1
                if n > 0 and current_state[py[i]][x] == player:
                    action = (0, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] + 1][px[i] + 1] == opponent:
                x = px[i] + 1
                y = py[i] + 1
                n = 0
                while current_state[y][x] == opponent:
                    x += 1
                    y += 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (1, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] + 1][px[i]] == opponent:
                y = py[i] + 1
                n = 0
                while current_state[y][px[i]] == opponent:
                    y += 1
                    n += 1
                if n > 0 and current_state[y][px[i]] == player:
                    action = (2, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] + 1][px[i] - 1] == opponent:
                y = py[i] - 1
                x = px[i] + 1
                n = 0
                while current_state[y][x] == opponent:
                    x -= 1
                    y += 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (3, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i]][px[i] - 1] == opponent:
                x = px[i] - 1
                n = 0
                while current_state[py[i]][x] == opponent:
                    x -= 1
                    n += 1
                if n > 0 and current_state[py[i]][x] == player:
                    action = (4, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] - 1][px[i] - 1] == opponent:
                x = px[i] - 1
                y = py[i] - 1
                n = 0
                while current_state[y, x] == opponent:
                    x -= 1
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (5, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] - 1][px[i]] == opponent:
                y = py[i] - 1
                n = 0
                while current_state[y][px[i]] == opponent:
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][px[i]] == player:
                    action = (6, n, px[i], py[i])
                    legal_actions.append(action)
            if current_state[py[i] - 1][px[i] + 1] == opponent:
                x = px[i] + 1
                y = py[i] - 1
                n = 0
                while current_state[y][x] == opponent:
                    x += 1
                    y -= 1
                    n += 1
                if n > 0 and current_state[y][x] == player:
                    action = (7, n, px[i], py[i])
                    legal_actions.append(action)
        print("player", player)
        print("legal_actions",legal_actions)
        return legal_actions

    def is_game_over(self):

        empty_case = np.where(self.state == 0)
        if np.size(empty_case) == 0:
            return True
        else:
            legal_action = self.get_legal_actions()
            if np.size(legal_action) == 0:
                self.player = 3 - self.player
                if np.size(legal_action) == 0:
                    return True #modif ici!
                else : 
                    self.player = 3 - self.player
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
        coord_y = action[3] # (7, n, px[i], py[i])
        n = action[1]
        current_state = self.state
        directions = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1), 4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1)}
        opponent = 3 - self.player

        current_state[coord_y][coord_x] = self.player
        xn, yn = directions[action[0]]
        for i in range(n):
            coord_x = coord_x + xn
            coord_y = coord_y + yn
            if current_state[coord_y][coord_x] == opponent:
                current_state[coord_y][coord_x] = self.player
        self.state = current_state
        print(self.state)
        return self.state
