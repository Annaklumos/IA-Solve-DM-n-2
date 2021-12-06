import numpy as np

class State:
    def __init__(self, first_state):
        self.state = first_state
        self.player = 1
        self.player_best_action = 1

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

        player_1 = np.count_nonzero(self.state == self.player)
        player_2 = np.count_nonzero(self.state.state == (3 - self.player))
        print(player_2)
        if player_1 > player_2:
            return 1
        elif player_1 < player_2:
            return -1
        elif player_1 == player_2:
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

        self.player = 3 - self.player

        return self.state

