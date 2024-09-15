# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from AdvancedPlayer import AdvancedPlayer
# from Constants import PLAYER_NAME_A, BLACK, PLAYER_NAME_B, QUEEN_MULTIPLIER, NOT_OVER_YET, DQN_OB_PATH, TIE
# from Move import Move
# from State import State
# from networks_helper.func_helper import index_to_square, square_to_index
# from collections import deque
#
#
# class DQN_Model(nn.Module):
#     def __init__(self, action_size):
#         super(DQN_Model, self).__init__()
#         self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 4 input channels
#         self.bn1 = nn.BatchNorm2d(64)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.fc1 = nn.Linear(256 * 8 * 8, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, action_size)  # Q-value output
#
#     def forward(self, x):
#         x = x.view(-1, 4, 8, 8)  # Reshape input to [batch, channels, height, width]
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = x.view(-1, 256 * 8 * 8)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         q_values = self.fc3(x)
#         return q_values  # Only return Q-values
#
#
# def action_index_to_move(action_index: int, state: State) -> Move:
#     from_index = action_index // 32
#     to_index = action_index % 32
#     from_loc = index_to_square(from_index)
#     to_loc = index_to_square(to_index)
#     if from_loc is None or to_loc is None:
#         return None
#     # Find the move in the state's possible moves
#     for move in state.find_all_moves():
#         if (move.get_piece_moved().get_location() == from_loc and
#             move.get_destination() == to_loc):
#             return move
#     return None  # Invalid move
#
#
# def move_to_action_index(move: Move) -> int:
#     from_loc = move.get_piece_moved().get_location()
#     to_loc = move.get_destination()
#     from_index = square_to_index(from_loc[0], from_loc[1])
#     to_index = square_to_index(to_loc[0], to_loc[1])
#     if from_index is None or to_index is None:
#         return None
#     action_index = from_index * 32 + to_index
#     return action_index
#
#
# def get_valid_action_indices(state: State):
#     moves = state.find_all_moves()
#     action_indices = []
#     for move in moves:
#         action_index = move_to_action_index(move)
#         if action_index is not None:
#             action_indices.append(action_index)
#     return action_indices
#
#
# class DQN_Player(AdvancedPlayer):
#     rl_update = True
#
#     def __init__(self, color: int) -> None:
#         super().__init__(color)
#         self.action_size = 32 * 32  # Total possible moves
#         self.game_history = deque(maxlen=10000)  # Fixed-size replay buffer
#         self.gamma = 0.99
#         self.epsilon = 1.0
#         self.epsilon_min = 0.01
#         self.epsilon_decay = 0.9995
#         self.learning_rate = 0.0005
#         self.batch_size = 64  # Increased batch size for better learning
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = DQN_Model(self.action_size).to(self.device)
#         self.target_model = DQN_Model(self.action_size).to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.mse_loss = nn.MSELoss()
#         self.update_counter = 0
#         self.target_update_freq = 1000  # Update target network every 1000 steps
#         player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
#         self.f_name = DQN_OB_PATH(f"{player_name}_{self.device}")
#         self.load_object()
#
#     def preprocess_state(self, state):
#         # Convert the state to a 4-channel 8x8 tensor
#         board = np.array(state.get_board_list())
#
#         channel1 = np.where(board == self.color, 1, 0)  # Player's regular pieces
#         channel2 = np.where(board == -self.color, 1, 0)  # Opponent's regular pieces
#         channel3 = np.where(board == self.color * QUEEN_MULTIPLIER, 1, 0)  # Player's queens
#         channel4 = np.where(board == -self.color * QUEEN_MULTIPLIER, 1, 0)  # Opponent's queens
#
#         processed_state = np.stack([channel1, channel2, channel3, channel4], axis=0)
#         return torch.FloatTensor(processed_state).unsqueeze(0)
#
#     def act(self, state: State) -> int:
#         valid_action_indices = get_valid_action_indices(state)
#         if not valid_action_indices:
#             return None  # No valid actions
#
#         if np.random.rand() <= self.epsilon:
#             action_index = random.choice(valid_action_indices)
#         # else:
#             self.model.eval()  # Set model to evaluation mode
#             state_tensor = self.preprocess_state(state).to(self.device)
#             with torch.no_grad():
#                 q_values = self.model(state_tensor).squeeze()
#             self.model.train()  # Set model back to training mode
#             # Create an action mask
#             action_mask = torch.full((self.action_size,), -float('inf'), device=self.device)
#             action_mask[valid_action_indices] = 0
#             masked_q_values = q_values + action_mask
#             action_index = torch.argmax(masked_q_values).item()
#         return action_index
#
#     def update_player(self, winner):
#         if not self.rl_update:
#             return
#
#         self.target_model.load_state_dict(self.model.state_dict())
#
#         reward = self.get_final_reward(winner)
#         for state, action, _, _, _ in reversed(self.game_history[-10:]):
#             state_tensor = self.preprocess_state(state).to(self.device)
#             action_tensor = torch.LongTensor([action]).unsqueeze(0).to(self.device)
#
#             current_q, _ = self.model(state_tensor)
#             target = torch.tensor([[reward]]).float().to(self.device)
#
#             loss = self.mse_loss(current_q.gather(1, action_tensor), target)
#
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
#
#             reward *= self.gamma
#
#     def make_move(self, state: State) -> State:
#         action_index = self.act(state)
#         if action_index is None:
#             # No valid moves, game over
#             return state
#
#         move = action_index_to_move(action_index, state)
#         if move is None:
#             # Should not happen if masking is correct
#             return state
#
#         next_state = state.next_state(move)
#         reward = self.reward_function(state, move, next_state)
#         is_done = next_state.is_over() != NOT_OVER_YET
#         self._remember(state, action_index, reward, next_state, is_done)
#
#         if len(self.game_history) > self.batch_size:
#             self.replay(self.batch_size)
#
#         # Update epsilon
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#         return next_state
#
#     def _remember(self, state, action, reward, next_state, done):
#         self.game_history.append((state, action, reward, next_state, done))
#
#     def replay(self, batch_size):
#         minibatch = random.sample(self.game_history, batch_size)
#         states, actions, rewards, next_states, dones = zip(*minibatch)
#         states = torch.cat([self.preprocess_state(s) for s in states]).to(self.device)
#         next_states_tensors = torch.cat([self.preprocess_state(s) for s in next_states]).to(self.device)
#         actions = torch.LongTensor(actions).to(self.device)
#         rewards = torch.FloatTensor(rewards).to(self.device)
#         dones = torch.FloatTensor(dones).to(self.device)
#         current_q_values = self.model(states)
#         next_q_values = self.target_model(next_states_tensors).detach()
#         batch_size = len(next_states)
#         action_mask = torch.full((batch_size, self.action_size), -float('inf'), device=self.device)
#         for i in range(batch_size):
#             valid_actions = get_valid_action_indices(next_states[i])
#             if valid_actions:
#                 action_mask[i, valid_actions] = 0
#         masked_next_q_values = next_q_values + action_mask
#         max_next_q_values, _ = masked_next_q_values.max(dim=1)
#         targets = rewards + self.gamma * max_next_q_values * (1 - dones)
#         predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
#         loss = self.mse_loss(predicted_q_values, targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#         self.optimizer.step()
#         self.update_counter += 1
#         if self.update_counter % self.target_update_freq == 0:
#             self.target_model.load_state_dict(self.model.state_dict())
#
#     def get_final_reward(self, winner):
#         """ Returns the final reward based on the game outcome."""
#         if winner == TIE:
#             return 0  # Neutral reward for a tie
#         elif winner == self.color:
#             return 100  # The last player is the opponent; we lost
#         else:
#             return -100  # We won
#
#     def reward_function(self, state: State, move: Move, next_state: State) -> float:
#         """
#         The complex reward function takes into account the capture of pieces, promotion to queen and protection of pieces.
#         """
#         reward = 0
#
#         # Capture tools - reward for it
#         opponent_pieces_before = np.sum(np.array(state.get_board_list()) == -self.color)
#         opponent_pieces_after = np.sum(np.array(next_state.get_board_list()) == -self.color)
#         if opponent_pieces_before - opponent_pieces_after > 0:
#             reward += (opponent_pieces_before - opponent_pieces_after) * 10
#
#         # Promotion to queen - providing a bonus for turning a tool into a queen
#         player_queens_before = np.sum(np.array(state.get_board_list()) == self.color * QUEEN_MULTIPLIER)
#         player_queens_after = np.sum(np.array(next_state.get_board_list()) == self.color * QUEEN_MULTIPLIER)
#
#         if player_queens_after - player_queens_before > 0:
#             reward += 5
#
#         # Reward for achieving a strategic location
#         if move.get_destination()[1] in [0, 7]:
#             reward += 2
#
#         # Loss of tools - punishment for that
#
#         all_opponent_moves = next_state.find_all_moves()
#         tmp_reward = 0
#         if all_opponent_moves:
#             for opponent_move in all_opponent_moves:
#                 for piece in opponent_move.pieces_eaten:
#                     tmp_reward -= 10 if piece.queen else 5  # Penalty for any lost tool
#             reward -= (tmp_reward / len(all_opponent_moves))
#
#         return reward
#
#     def load_object(self):
#         if os.path.isfile(self.f_name):
#             print(f"DQN weights file loads from path: {self.f_name}")
#             self.model.load_state_dict(torch.load(self.f_name))
#             self.target_model.load_state_dict(self.model.state_dict())
#         else:
#             print("No existing model found. Starting from scratch.")
#             self.target_model.load_state_dict(self.model.state_dict())
#
#     def save_object(self):
#         print(f"DQN weights file saved to path: {self.f_name}")
#         AdvancedPlayer.rename_old_state_file(self.f_name)
#         torch.save(self.model.state_dict(), self.f_name)
