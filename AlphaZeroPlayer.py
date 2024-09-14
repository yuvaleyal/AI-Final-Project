# AlphaZeroPlayer.py
import os
import torch
from torch import optim
import torch.nn.functional as F
from AdvancedPlayer import AdvancedPlayer
from Constants import OBJECTS_DIR, PLAYER_NAME_A, BLACK, PLAYER_NAME_B, AlphaZeroNET_OB_PATH
from State import State
from networks_helper.func_helper import state_to_tensor, augment_data
from networks_helper.mcts import MCTS
from networks_helper.nets import PolicyValueNet
from collections import deque
import random


class AlphaZeroPlayer(AdvancedPlayer):
    def __init__(self, color):
        super().__init__(color)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_value_net = PolicyValueNet().to(self.device)
        self.optimizer = optim.AdamW(self.policy_value_net.parameters(), lr=0.0003, weight_decay=1e-4)
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        file_name_unique = f"{player_name}_{str(self.device)}"
        self.num_simulations = 1600
        self.c_puct = 1.0
        self.mcts = MCTS(self.policy_value_net, self.num_simulations, self.c_puct, self.device)
        self.buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.alpha_zero_net_path = AlphaZeroNET_OB_PATH(file_name_unique)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.elo_rating = 1000
        self.game_length = 0

        self.load_object()

    def make_move(self, state: State) -> State:
        # Use MCTS to select a move
        # self.mcts = MCTS(self.policy_value_net, self.num_simulations, self.c_puct, self.device)
        move, pi = self.mcts.get_action(state, temp=1e-3)
        self.buffer.append((state_to_tensor(state), pi, None))  # Winner will be updated later
        self.game_length += 1
        next_state = state.next_state(move)
        return next_state

    def update_player(self, winner):
        # Update the winner information in the buffer
        reward = 1 if winner == self.color else -1 if winner == -self.color else 0
        # Update the last game entries with the reward
        num_entries = len(self.buffer)
        for idx in range(num_entries - self.game_length, num_entries):
            state_tensor, pi, _ = self.buffer[idx]
            self.buffer[idx] = (state_tensor, pi, reward)
        self.game_length = 0
        # Update neural network with self-play data
        if len(self.buffer) >= self.batch_size:
            self.train_network()

        # Save the network weights periodically
        if len(self.buffer) % (self.batch_size * 10) == 0:
            self.save_object()
        self.mcts = MCTS(self.policy_value_net, self.num_simulations, self.c_puct, self.device)

    def train_network(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        state_batch = torch.stack([s for s, _, _ in mini_batch]).to(self.device)
        mcts_probs_batch = torch.stack([mp for _, mp, _ in mini_batch]).to(self.device)
        winner_batch = torch.tensor([w for _, _, w in mini_batch], dtype=torch.float32).to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()
        # Forward + backward + optimize
        log_probs, values = self.policy_value_net(state_batch)
        value_loss = F.mse_loss(values.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.lr_scheduler.step()

    def save_object(self):
        AdvancedPlayer.rename_old_state_file(self.alpha_zero_net_path)
        # Save the neural network's weights
        torch.save(self.policy_value_net.state_dict(), self.alpha_zero_net_path)
        print(f"AZ weights file saved to path: {self.alpha_zero_net_path}")

    def load_object(self):
        if not os.path.exists(OBJECTS_DIR):
            os.makedirs(OBJECTS_DIR)
        print(f"AZ weights file load from path: {self.alpha_zero_net_path}")
        if os.path.isfile(self.alpha_zero_net_path):
            self.policy_value_net.load_state_dict(
                torch.load(self.alpha_zero_net_path, map_location=self.device, weights_only=True))
        else:
            print("No existing model found. Starting from scratch.")
