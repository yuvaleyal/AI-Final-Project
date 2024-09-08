from Player import Player
from State import State
import numpy as np
import random


class ReinforcementPlayer(Player):
    def __init__(self, color: int) -> None:
        super().__init__(color)
        self.g_agent = QLearningAgent(8*8, 10)

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        move = self.g_agent.choose_action(state)

        reward = calculate_reward(board, move, self.color)
        new_state = state.next_state(move)
        self.g_agent.learn(state, move, reward, new_state)
        return new_state


def calculate_reward(board, move, player):
    """
    A function that calculates the reward for a certain movement on the board.
    """
    # If the player eats an opponent's tool, the reward will be positive
    if move['type'] == 'capture':
        return 10
    # If the player has not performed any significant action, there is no reward
    return 0


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_size, action_size))  # The Q table

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Exploration, random action
        return np.argmax(self.q_table[state])  # Utilization, operation with maximum Q value

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

        # exploratory probability update
        if self.epsilon > 0.1:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename):
        np.save(filename, self.q_table)  # Saving the Q table to NPY file

    def load_q_table(self, filename):
        self.q_table = np.load(filename)  # Loading a Q table from an NPY file
