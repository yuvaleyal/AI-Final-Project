import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from AdvancedPlayer import AdvancedPlayer
from Constants import PLAYER_NAME_A, BLACK, PLAYER_NAME_B, QUEEN_MULTIPLIER, NOT_OVER_YET, DQN_OB_PATH
from Move import Move
from State import State
from networks_helper.func_helper import index_to_square, square_to_index


class DQN_Model(nn.Module):
    def __init__(self, action_size):
        super(DQN_Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 4 input channels
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size)  # Q-value output

    def forward(self, x):
        x = x.view(-1, 4, 8, 8)  # Reshape input to [batch, channels, height, width]
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values  # Only return Q-values


def action_index_to_move(action_index: int, state: State) -> Move:
    from_index = action_index // 32
    to_index = action_index % 32
    from_loc = index_to_square(from_index)
    to_loc = index_to_square(to_index)
    if from_loc is None or to_loc is None:
        return None
    # Find the move in the state's possible moves
    for move in state.find_all_moves():
        if (move.get_piece_moved().get_location() == from_loc and
                move.get_destination() == to_loc):
            return move
    return None  # Invalid move


def move_to_action_index(move: Move) -> int:
    from_loc = move.get_piece_moved().get_location()
    to_loc = move.get_destination()
    from_index = square_to_index(from_loc[0], from_loc[1])
    to_index = square_to_index(to_loc[0], to_loc[1])
    if from_index is None or to_index is None:
        return None
    action_index = from_index * 32 + to_index
    return action_index


def get_valid_action_indices(state: State):
    moves = state.find_all_moves()
    action_indices = []
    for move in moves:
        action_index = move_to_action_index(move)
        if action_index is not None:
            action_indices.append(action_index)
    return action_indices


class RLDNNPlayer(AdvancedPlayer):
    rl_update = True

    def __init__(self, color: int) -> None:
        super().__init__(color)
        self.action_size = 32 * 32  # Total possible moves
        self.game_history = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN_Model(self.action_size).to(self.device)
        self.target_model = DQN_Model(self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()
        self.update_counter = 0
        self.target_update_freq = 1000  # Update target network every 1000 steps
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        self.f_name = DQN_OB_PATH(f"{player_name}_{self.device}")

    def preprocess_state(self, state):
        # Convert the state to a 4-channel 8x8 tensor
        board = np.array(state.get_board_list())

        channel1 = np.where(board == self.color, 1, 0)  # Player's regular pieces
        channel2 = np.where(board == -self.color, 1, 0)  # Opponent's regular pieces
        channel3 = np.where(board == self.color * QUEEN_MULTIPLIER, 1, 0)  # Player's queens
        channel4 = np.where(board == -self.color * QUEEN_MULTIPLIER, 1, 0)  # Opponent's queens

        processed_state = np.stack([channel1, channel2, channel3, channel4], axis=0)
        return torch.FloatTensor(processed_state).unsqueeze(0)

    def act(self, state: State) -> int:
        state_tensor = self.preprocess_state(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze()
        valid_action_indices = get_valid_action_indices(state)
        if len(valid_action_indices) == 0:
            return None  # No valid actions
        valid_q_values = q_values[valid_action_indices]
        if np.random.rand() <= self.epsilon:
            action_index = random.choice(valid_action_indices)
        else:
            max_index = torch.argmax(valid_q_values).item()
            action_index = valid_action_indices[max_index]
        return action_index

    def update_player(self, winner):
        pass

    def make_move(self, state: State) -> State:
        action_index = self.act(state)
        if action_index is None:
            # No valid moves, game over
            return state

        move = action_index_to_move(action_index, state)
        if move is None:
            # Should not happen if masking is correct
            return state

        next_state = state.next_state(move)
        reward = self.reward_function(state, next_state)
        is_done = next_state.is_over() != NOT_OVER_YET
        self._remember(state, action_index, reward, next_state, is_done)

        if len(self.game_history) > self.batch_size:
            self.replay(self.batch_size)

        return next_state

    def _remember(self, state, action, reward, next_state, done):
        self.game_history.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.game_history, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.cat([self.preprocess_state(s) for s in states]).to(self.device)
        next_states = torch.cat([self.preprocess_state(s) for s in next_states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()

        # Mask invalid actions in next states
        max_next_q_values = []
        for i in range(len(next_states)):
            valid_actions = self.get_valid_action_indices_from_state_tensor(next_states[i])
            if len(valid_actions) > 0:
                max_q = torch.max(next_q_values[i][valid_actions])
            else:
                max_q = torch.tensor(0.0).to(self.device)
            max_next_q_values.append(max_q)
        max_next_q_values = torch.stack(max_next_q_values)

        # Compute targets
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Gather the Q-values for the actions taken
        predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.mse_loss(predicted_q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def reward_function(self, state: State, next_state: State):
        if next_state.is_over() != NOT_OVER_YET:
            winner = next_state.is_over()
            if winner == self.color:
                return 100  # Win
            elif winner == -self.color:
                return -100  # Loss
            else:
                return 0  # Tie
        # Intermediate reward
        reward = 0
        # Encourage capturing opponent's pieces
        opponent_pieces_before = np.sum(np.array(state.get_board_list()) == -self.color)
        opponent_pieces_after = np.sum(np.array(next_state.get_board_list()) == -self.color)
        reward += (opponent_pieces_before - opponent_pieces_after) * 10
        # Penalize losing own pieces
        player_pieces_before = np.sum(np.array(state.get_board_list()) == self.color)
        player_pieces_after = np.sum(np.array(next_state.get_board_list()) == self.color)
        reward -= (player_pieces_before - player_pieces_after) * 10
        # Encourage promotion to queen
        player_queens_before = np.sum(np.array(state.get_board_list()) == self.color * QUEEN_MULTIPLIER)
        player_queens_after = np.sum(np.array(next_state.get_board_list()) == self.color * QUEEN_MULTIPLIER)
        reward += (player_queens_after - player_queens_before) * 5
        return reward

    def load_object(self):
        if os.path.isfile(self.f_name):
            self.model.load_state_dict(torch.load(self.f_name))
            self.target_model.load_state_dict(self.model.state_dict())

    def save_object(self):
        AdvancedPlayer.rename_old_state_file(self.f_name)
        torch.save(self.model.state_dict(), self.f_name)

    # Helper methods

    def get_valid_action_indices_from_state_tensor(self, state_tensor):
        return list(range(self.action_size))
