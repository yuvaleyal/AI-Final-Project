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
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        print(f'Player name: {player_name}')
        self.f_name = Q_Learning_OB_PATH(player_name)
        self.load_object()


    def load_object(self):
        if os.path.isfile(self.f_name):
            print(f"RL weights file loads from path: {self.f_name}")
            with open(Q_Learning_OB_PATH(self.f_name), 'rb') as file:
                self.q_agent = pickle.load(file)
        else:
            self.q_agent = QLearning(alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01)

    def save_object(self):
        AdvancedPlayer.rename_old_state_file(self.f_name)
        with open(self.f_name, 'wb') as file:
            pickle.dump(self.q_agent, file)
        print(f"RL weights file saved to path: {self.f_name}")

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        move = self.q_agent.choose_action(state, options, self.color)
        # Store the experience for learning
        if ReinforcementPlayer.rl_update:
            self.q_agent.store_experience(state, move)
        new_state = state.next_state(move)
        return new_state

    def update_player(self, winner):
        if ReinforcementPlayer.rl_update:
            self.q_agent.learn(winner)
            # Decay epsilon
            self.q_agent.epsilon = max(self.q_agent.epsilon_min, self.q_agent.epsilon * self.q_agent.epsilon_decay)


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
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01) -> None:
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
            # In case multiple actions have the same max Q-value
            best_actions = [move for move, q in zip(options, q_values) if q == max_q]
            return random.choice(best_actions)  # Exploit

    def store_experience(self, state: State, action: Move):
        self.experiences.append((state, action))

    def learn(self, winner):
        """ Update Q-values at the end of the game."""
        reward = self.get_final_reward(winner)
        for idx in reversed(range(len(self.experiences))):
            state, action = self.experiences[idx]
            state_key = state_to_key(state)
            action_key = move_to_key(action)
            if idx < len(self.experiences) - 1:
                next_state, _ = self.experiences[idx + 1]
                next_state_key = state_to_key(next_state)
                next_q_values = [self.q_table.get((next_state_key, a_key), 0) for a_key in
                                 get_possible_action_keys(next_state)]
                max_next_q = max(next_q_values) if next_q_values else 0
            else:
                max_next_q = 0  # Terminal state

            # Q-learning update
            current_q = self.q_table.get((state_key, action_key), 0)
            updated_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[(state_key, action_key)] = updated_q
            reward = 0

        self.experiences = []

    def get_final_reward(self, winner):
        """ Returns the final reward based on the game outcome."""
        if winner == TIE:
            return 0  # Neutral reward for a tie
        elif winner == self.experiences[-1][0].last_player:
            return -1  # The last player is the opponent; we lost
        else:
            return 1  # We won
