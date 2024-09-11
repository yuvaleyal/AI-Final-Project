from AdvancedPlayer import AdvancedPlayer
from Constants import *
from Move import Move
from Pieces import RegularPiece, QueenPiece, Piece
from State import State
import numpy as np
import random
import pickle
import os.path

choose_random = "choose_random"
choose_q = "choose_q"
max_next = "max_next"


class ReinforcementPlayer(AdvancedPlayer):
    rl_update = True

    def __init__(self, color: int) -> None:
        super().__init__(color)
        self.game_history = []
        self.q_agent = None

    def load_object(self):
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        if os.path.isfile(Q_Learning_OB_PATH(player_name)):
            with open(Q_Learning_OB_PATH(player_name), 'rb') as file:
                self.q_agent = pickle.load(file)
        else:
            self.q_agent = QLearning(8 * 8)

    def save_object(self):
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        AdvancedPlayer.rename_old_state_file(Q_Learning_OB_PATH(player_name))
        with open(Q_Learning_OB_PATH(player_name), 'wb') as file:
            pickle.dump(self.q_agent, file)

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        move = self.q_agent.choose_action(state, options)
        new_state = state.next_state(move)
        if ReinforcementPlayer.rl_update:
            self.q_agent.update_q_value(state, move, new_state)
        return new_state

    def update_player(self, winner):
        self.q_agent.epsilon = max(self.q_agent.epsilon_min, self.q_agent.epsilon * self.q_agent.epsilon_decay)


class QLearning:

    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01) -> None:

        self.q_table = dict()
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state: State, options: list[Move]) -> Move:
        """ Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.choice(options)  # Explore
        else:
            return max(options, key=lambda move: self.q_table.get((state, move), 0))  # Exploit

    def best_possible_operator(self, state, action, reward, next_state):
        """
        Best Possible Operator (BPO) for updating Q-values more optimally.
        This function implements the BPO approach, which considers both current action
        and future expected reward to update Q-values in a more stable manner.
        """
        # If next_state or current state-action pair does not exist, initialize it
        next_moves = next_state.find_all_moves()
        for move in next_moves:
            if (next_state, move) not in self.q_table:
                self.q_table[(next_state, move)] = 0  # Initialize future state
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0  # Initialize Q-value for state-action pair

        score_moves =[self.q_table.get((next_state, action), 0) for action in next_state.find_all_moves()]
        score_moves += [0]
        best_future_q = max(score_moves)
        updated_q_value = (1 - self.alpha) * self.q_table.get((state, action), 0) + self.alpha * (
            reward + self.gamma * best_future_q)
        return updated_q_value

    def update_q_value(self, state, action, next_state):
        """ Update Q-value using Best Possible Operator (BPO)."""
        current_player = -state.last_player
        reward = QLearning.calculate_reward(next_state, current_player)
        self.q_table[state, action] = self.best_possible_operator(state, action, reward, next_state)

    @staticmethod
    def calculate_reward(opponent_state: State, the_player: int):
        if not opponent_state.find_all_moves():
            # the_player win
            return 15
        opponent_moves = opponent_state.find_all_moves()
        reward = 0
        for opponent_move in opponent_moves:
            the_player_state = opponent_state.generate_successor(opponent_move)
            if not the_player_state.find_all_moves():
                reward -= 15
            else:
                opponent_pieces = np.sum(np.array(the_player_state.get_board_list()).flatten() == -the_player)
                player_pieces = np.sum(np.array(the_player_state.get_board_list()).flatten() == the_player)
                opponent_queen_pieces = np.sum(
                    np.array(the_player_state.get_board_list()).flatten() == (-the_player) * 10)
                player_queen_pieces = np.sum(np.array(the_player_state.get_board_list()).flatten() == the_player * 10)
                reward += (1 * (player_pieces - opponent_pieces) + 3 * (player_queen_pieces - opponent_queen_pieces))

        return reward / len(opponent_moves)
