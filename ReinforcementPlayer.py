import os
import pickle
import random
import numpy as np
from AdvancedPlayer import AdvancedPlayer
from Constants import *
from Move import Move
from State import State


class ReinforcementPlayer(AdvancedPlayer):
    rl_update = True

    def __init__(self, color: int) -> None:
        super().__init__(color)
        self.game_history = []
        self.q_agent = None
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        print(f'Player name: {player_name}')
        self.f_name = Q_Learning_OB_PATH(player_name)
        self.load_object()

    def load_object(self):
        if os.path.isfile(self.f_name):
            print(f"RL weights file loads from path: {self.f_name}")
            with open(self.f_name, 'rb') as file:
                self.q_agent = pickle.load(file)
        else:
            self.q_agent = QLearning(color=self.color, alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon,
                                     epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min)

    def save_object(self):
        AdvancedPlayer.rename_old_state_file(self.f_name)
        with open(self.f_name, 'wb') as file:
            pickle.dump(self.q_agent, file)
        print(f"RL weights file saved to path: {self.f_name}")

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        move = self.q_agent.choose_action(state, options, self.color)

        # שמירת ניסיון על מנת ללמוד מיד לאחר המהלך
        if ReinforcementPlayer.rl_update:
            new_state = state.next_state(move)
            reward = self.reward_function(state, move, new_state)
            self.q_agent.store_experience(state, move, reward, new_state)
            self.q_agent.learn_incrementally(state, move, reward, new_state)  # למידה מיד לאחר המהלך

        else:
            new_state = state.next_state(move)

        return new_state

    def update_player(self, winner):
        if ReinforcementPlayer.rl_update:
            self.q_agent.learn_from_game(winner)
            # Decay epsilon
            self.q_agent.epsilon = max(self.q_agent.epsilon_min, self.q_agent.epsilon * self.q_agent.epsilon_decay)

    def reward_function(self, state: State, move:Move, next_state:State) -> float:
        """
        The complex reward function takes into account the capture of pieces, promotion to queen and protection of pieces.
        """
        reward = 0

        # Capture tools - reward for it
        opponent_pieces_before = np.sum(np.array(state.get_board_list()) == -self.color)
        opponent_pieces_after = np.sum(np.array(next_state.get_board_list()) == -self.color)
        if opponent_pieces_before - opponent_pieces_after > 0:
            reward += (opponent_pieces_before - opponent_pieces_after) * 10

        # Promotion to queen - providing a bonus for turning a tool into a queen
        player_queens_before = np.sum(np.array(state.get_board_list()) == self.color * QUEEN_MULTIPLIER)
        player_queens_after = np.sum(np.array(next_state.get_board_list()) == self.color * QUEEN_MULTIPLIER)

        if player_queens_after - player_queens_before > 0:
            reward += 5

        # Reward for achieving a strategic location
        if move.get_destination()[1] in [0, 7]:
            reward += 2

        # Loss of tools - punishment for that

        all_opponent_moves = next_state.find_all_moves()
        tmp_reward = 0
        if all_opponent_moves:
            for opponent_move in all_opponent_moves:
                for piece in opponent_move.pieces_eaten:
                    tmp_reward -= 10 if piece.queen else 5 # Penalty for any lost tool
            reward -= (tmp_reward/len(all_opponent_moves))

        return reward


# QLearning.py

def move_to_key(move: Move):
    """ Convert move to a hashable key."""
    piece_loc = move.get_piece_moved().get_location()
    dest = move.get_destination()
    return piece_loc, dest


def get_possible_action_keys(state: State):
    moves = state.find_all_moves()
    return [move_to_key(move) for move in moves]


def state_to_key(state: State):
    """ Convert state to a hashable key."""
    board_list = state.get_board_list()
    # Flatten the board to a tuple
    return tuple(item for row in board_list for item in row)


class QLearning:
    def __init__(self, color, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01) -> None:
        self.color = color
        print(self.color)
        self.q_table = dict()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.experiences = []  # To store state, action, reward, next_state transitions

    def choose_action(self, state: State, options: list[Move], player_color: int) -> Move:
        """ Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.choice(options)  # Explore
        else:
            q_values = [self.q_table.get((state_to_key(state), move_to_key(move)), 0) for move in options]
            max_q = max(q_values)
            best_actions = [move for move, q in zip(options, q_values) if q == max_q]
            return random.choice(best_actions)  # Exploit

    def store_experience(self, state: State, action: Move, reward: float, next_state: State):
        self.experiences.append((state, action, reward, next_state))

    def learn_incrementally(self, state: State, action: Move, reward: float, next_state: State):
        """
        """
        state_key = state_to_key(state)
        action_key = move_to_key(action)
        next_state_key = state_to_key(next_state)

        next_q_values = [self.q_table.get((next_state_key, a_key), 0) for a_key in get_possible_action_keys(next_state)]
        max_next_q = max(next_q_values) if next_q_values else 0

        current_q = self.q_table.get((state_key, action_key), 0)
        updated_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_key, action_key)] = updated_q

    def learn_from_game(self, winner):
        """
        Learning based on game results. Q update at the end of the game.
        """
        reward = self.get_final_reward(winner)
        for idx in reversed(range(len(self.experiences))):
            state, action, _, next_state = self.experiences[idx]
            self.learn_incrementally(state, action, reward, next_state)
            reward = 0  # Reward is reset to continue the experience

        self.experiences = []

    def get_final_reward(self, winner):
        """ Returns the final reward based on the game outcome."""
        if winner == TIE:
            return 0  # Neutral reward for a tie
        elif winner == (-self.color):
            return -50  # The last player is the opponent; we lost
        else:
            return 50  # We won
