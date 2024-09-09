from Constants import *
from Pieces import RegularPiece, QueenPiece
from Player import Player
from State import State
import numpy as np
import random
import pickle
import os.path
from datetime import datetime


class ReinforcementPlayer(Player):
    def __init__(self, color: int) -> None:
        super().__init__(color)
        self.q_agent = None

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        vec_state = tuple(np.array(state.get_board_list()).flatten())
        if not options:
            move = None
            reward = -1.0
        else:
            move = self.q_agent.choose_action(vec_state, options)
            reward = calculate_reward(move, self.color)
        new_state = state.next_state(move)
        other_options = new_state.find_all_moves()
        vec_new_state = tuple(np.array(new_state.get_board_list()).flatten())
        self.q_agent.update_q_value(vec_state, move, reward, vec_new_state, other_options)
        return new_state

    def load_object(self, player_name):
        if os.path.isfile(Q_Learning_OB_PATH(player_name)):
            with open(Q_Learning_OB_PATH(player_name), 'rb') as file:
                self.q_agent = pickle.load(file)
        else:
            self.q_agent = QLearningAgent(8 * 8)

    def save_object(self, player_name, num_games: int):
        if os.path.isfile(Q_Learning_OB_PATH(player_name)):
            os.rename(Q_Learning_OB_PATH(player_name),
                      Q_Learning_OB_PATH(f"{player_name}-{datetime.now().timestamp()}"))
        with open(Q_Learning_OB_PATH(player_name), 'wb') as file:
            self.q_agent.num_games += num_games
            pickle.dump(self.q_agent, file)


# def calculate_reward(board, move, player):
def calculate_reward(move, player):
    """
    A function that calculates the reward for a certain movement on the board.
    """
    reward = 0.0
    for piece in move.get_pieces_eaten():
        if isinstance(piece, RegularPiece):
            reward += 2
        if isinstance(piece, QueenPiece):
            reward += 5

    # If the player eats an opponent's tool, the reward will be positive
    # if move['type'] == 'capture':
    #     return 10
    # print("get_pieces_eaten", move.get_pieces_eaten(), "reward", reward)
    return reward


class QLearningAgent:
    def __init__(self, state_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = {}
        self.num_games = 0

    def get_q_value(self, state, action):
        """
        Updating the Q value for a certain state and a certain action according to the reward and the next state.
        """
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, options):
        if (t:= np.random.rand()) <= self.epsilon:
            # print("random vs eps:", t, self.epsilon)
            return random.choice(options)  # Exploration, random action

        # Exploitation - finding the operation with the highest Q value from among the legal operations
        q_values = [self.get_q_value(state, action) for action in options]
        best_action = options[np.argmax(q_values)]

        return best_action

    def update_q_value(self, state, action, reward, next_state, legal_actions):
        """
         Updating the Q value for a certain state and a certain action according to the reward and the next state.
         legal_actions: the list of legal actions in the next state
        """
        # Calculate the current Q value
        current_q = self.get_q_value(state, action)

        # Finding the maximum Q value for the legal operations in the following state
        next_q_values = [self.get_q_value(next_state, a) for a in legal_actions]
        max_next_q = max(next_q_values) if next_q_values else 0.0  # If no valid actions, no value

        # Calculation of the reward target (TD Target)
        td_target = reward + self.gamma * max_next_q

        # Updating the Q value according to the Q-Learning equation
        new_q = current_q + self.alpha * (td_target - current_q)

        # Saving the updated value in the Q table
        self.q_table[(state, action)] = new_q

    def save_q_table(self, filename):
        # print(self.q_table)
        np.save(filename, self.q_table)  # Saving the Q table to NPY file

    def load_q_table(self, filename):
        self.q_table = np.load(filename, allow_pickle=True).item()
        # self.q_table = np.load(filename)  # Loading a Q table from an NPY file

    def decay_epsilon(self):
        """
        Decaying the epsilon value to reduce exploration and increase exploitation as learning progresses.
        """
        if self.epsilon > 0.1:
            print ("epsilon", self.epsilon)
            self.epsilon *= self.epsilon_decay
