import os
import torch
from torch import optim
import torch.nn.functional as F
from AdvancedPlayer import AdvancedPlayer
from Constants import OBJECTS_DIR, PLAYER_NAME_A, BLACK, PLAYER_NAME_B, AlphaZeroNET_OB_PATH
from State import State
from networks_helper.func_helper import state_to_tensor
from networks_helper.mcts import MCTS
from networks_helper.nets import PolicyValueNet


class AlphaZeroPlayer(AdvancedPlayer):
    def __init__(self, color):
        super().__init__(color)
        self.policy_value_net = PolicyValueNet()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        file_name_unique = f"{player_name}_{str(self.device)}"
        self.num_simulations = 800
        self.c_puct = 1.0
        self.mcts = None
        self.buffer = []  # For storing self-play data
        self.batch_size = 64
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_value_net.to(self.device)
        self.alpha_zero_net_path = AlphaZeroNET_OB_PATH(file_name_unique)

    def make_move(self, state: State) -> State:
        # Use MCTS to select a move
        self.mcts = MCTS(self.policy_value_net, self.num_simulations, self.c_puct)
        move = self.mcts.get_action(state)
        # Store the data for training
        self.buffer.append((state, self.mcts.get_policy_distribution(state)))
        # Apply the move
        next_state = state.generate_successor(move)
        return next_state

    def update_player(self, winner):
        # Update neural network with self-play data
        states, mcts_probs, winners = self.prepare_training_data(winner)
        dataset = list(zip(states, mcts_probs, winners))
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            state_batch = torch.stack([state_to_tensor(s) for s, _, _ in batch]).to(self.device)
            mcts_probs_batch = torch.tensor([mp for _, mp, _ in batch], dtype=torch.float32).to(self.device)
            winner_batch = torch.tensor([w for _, _, w in batch], dtype=torch.float32).to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            # Forward + backward + optimize
            log_probs, values = self.policy_value_net(state_batch)
            value_loss = F.mse_loss(values.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_probs, dim=1))
            loss = value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
        # Clear the buffer
        self.buffer = []

    def prepare_training_data(self, winner):
        # Prepare data from the buffer for training
        states = []
        mcts_probs = []
        winners = []
        for state, pi in self.buffer:
            states.append(state)
            mcts_probs.append(pi)
            winners.append(winner if state.last_player != self.color else -winner)
        return states, mcts_probs, winners

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
                torch.load(self.alpha_zero_net_path, weights_only=True, map_location=self.device))
