import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict

from AdvancedPlayer import AdvancedPlayer
from Constants import NOT_OVER_YET, AlphaZeroNET_OB_PATH, MCTS_OB_PATH, OBJECTS_DIR, PLAYER_NAME_A, PLAYER_NAME_B, BLACK
from State import State

POLICY_SIZE = 70


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.policy_head = nn.Linear(256, POLICY_SIZE)  # Output POLICY_SIZE possible moves
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 8 * 8 * 128)
        x = torch.relu(self.fc1(x))

        # Policy head - probabilities for all legal actions
        policy = torch.softmax(self.policy_head(x), dim=1)

        # Value head - estimated value of the board state
        value = torch.tanh(self.value_head(x))
        return policy, value


class MCTS:
    def __init__(self, color, n_simulations=800, c_puct=1.0):
        self.color = color
        self.model = None
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.Q = defaultdict(float)  # Action value Q(s, a) action-value estimates
        self.N = defaultdict(int)  # Visit count N(s, a)
        self.P = {}  # Prior probabilities from the policy network

        self.board_size = 8

    def set_model(self, model):
        self.model = model

    def reset_P(self):
        self.P = {}

    def ucb1(self, state: State, a, legal_moves):
        """Upper Confidence Bound calculation."""
        return (self.Q[(state, a)] + self.c_puct * self.P[state][a] *
                np.sqrt(sum(self.N[(state, move)] for move in legal_moves)) / (1 + self.N[(state, a)]))

    def simulate(self, state: State):
        """Simulate a single MCTS run."""
        path = []
        current_state = state

        while current_state.is_over() == NOT_OVER_YET:
            legal_moves = current_state.find_all_moves()

            # If the state is not expanded, expand using the neural network
            if current_state not in self.P:
                policy, value = self.model(torch.FloatTensor(current_state.get_board_list()).unsqueeze(0).unsqueeze(0))
                policy = policy.detach().numpy()[0]

                # Initialize the prior probabilities for this state
                self.P[current_state] = {move: policy[i] for i, move in enumerate(legal_moves)}
                # Initialize Q and N values
                for move in legal_moves:
                    if (current_state, move) not in self.Q:
                        self.Q[(current_state, move)] = 0
                        self.N[(current_state, move)] = 0
                return value

            # best_move = max(legal_moves,
            #                 key=lambda move: self.Q[(current_state, move)] + self.c_puct * self.P[current_state][move] *
            #                                  np.sqrt(sum(self.N[(current_state, a)] for a in legal_moves)) / (
            #                                      1 + self.N[(current_state, move)]))
            best_move = max(legal_moves, key=lambda move: self.ucb1(current_state, move, legal_moves))

            path.append((current_state, best_move))
            current_state = current_state.generate_successor(best_move)

        # Backpropagate the result
        reward = self.get_final_score(current_state)
        for state, move in path:
            self.N[(state, move)] += 1
            self.Q[(state, move)] += (reward - self.Q[(state, move)]) / self.N[(state, move)]

    def get_best_move(self, state: State):
        """Select the move with the highest visit count."""
        legal_moves = state.find_all_moves()
        move_visits = [self.N[(state, move)] for move in legal_moves]
        return legal_moves[np.argmax(move_visits)]

    def get_final_score(self, cur_state: State):
        winner = cur_state.is_over()
        reward_ = 0.0
        if self.color == winner:
            reward_ += 10
        elif self.color == -winner:
            reward_ -= 10
        return reward_

    def search(self, state):
        """Run MCTS and return the best move."""
        for _ in range(self.n_simulations):
            self.simulate(state)
        return self.get_best_move(state)


class AlphaZeroPlayer(AdvancedPlayer):
    def __init__(self, color, n_simulations=800, c_puct=1.0, min_visits=10):
        super().__init__(color)
        player_name = PLAYER_NAME_A if self.color == BLACK else PLAYER_NAME_B
        self.alpha_zero_net_path = AlphaZeroNET_OB_PATH(player_name)
        self.mcts_path = MCTS_OB_PATH(player_name)
        self.optimizer = None
        self.model = AlphaZeroNet()
        self.mcts = MCTS(color, n_simulations, c_puct)
        self.game_history = []
        self.min_visits = min_visits  # Minimum number of visits to save P(s, a)

    def make_move(self, state: State) -> State:
        move = self.mcts.search(state)
        self.game_history.append((state, move))
        new_state = state.next_state(move)
        return new_state

    def update_player(self, winner):
        """Plays multiple games between AlphaZero and a random player."""
        # Prepare training data (states, policies, values)
        self.model.train()
        states = [state for state, _ in self.game_history]
        policies = [self.mcts.P[state] for state, _ in self.game_history]
        rewards = [winner] * len(self.game_history)

        # Train the model
        for state, policy, reward in zip(states, policies, rewards):
            state_tensor = torch.FloatTensor(state.get_board_list()).unsqueeze(0).unsqueeze(0)
            policy_vec = convert_moves_dict2vec(policy)
            policy_tensor = torch.FloatTensor(policy_vec).unsqueeze(0)
            value_tensor = torch.FloatTensor([reward]).unsqueeze(0)

            self.optimizer.zero_grad()
            predicted_policy, predicted_value = self.model(state_tensor)

            policy_loss = torch.nn.functional.mse_loss(predicted_policy, policy_tensor)
            value_loss = torch.nn.functional.mse_loss(predicted_value, value_tensor)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()
        self.game_history = []
        # self.mcts.reset_P()

    def save_object(self):
        AdvancedPlayer.rename_old_state_file(self.alpha_zero_net_path)
        # Save the neural network's weights
        torch.save(self.model.state_dict(), self.alpha_zero_net_path)
        # to_save_P = {}
        # for state, actions in self.mcts.P.items():
        #     if any(self.mcts.N[(state, move)] > self.min_visits for move in actions):
        #         to_save_P[state] = actions
        # self.mcts.P = to_save_P
        # Save the MCTS fields
        mcts_fields = {
            'Q': self.mcts.Q,
            'N': self.mcts.N,
            # 'P': self.mcts.P
        }
        AdvancedPlayer.rename_old_state_file(self.mcts_path)
        with open(self.mcts_path, 'wb') as f:
            pickle.dump(mcts_fields, f)

    def load_object(self):
        if not os.path.exists(OBJECTS_DIR):
            os.makedirs(OBJECTS_DIR)
        if os.path.isfile(self.alpha_zero_net_path):
            self.model.load_state_dict(torch.load(self.alpha_zero_net_path, weights_only=True))
            self.model.eval()  # Set the model to evaluation mode
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # # Load the MCTS fields
        # if os.path.isfile(self.mcts_path):
        #     with open(self.mcts_path, 'rb') as f:
        #         loaded_mcts_fields = pickle.load(f)
        #     self.mcts.Q = loaded_mcts_fields['Q']
        #     self.mcts.N = loaded_mcts_fields['N']
        #     self.mcts.P = loaded_mcts_fields['P']
        self.mcts.set_model(self.model)


def convert_piece2vec(piece):
    type_p = 10 if piece.queen else 1
    policy = [piece.player, type_p, piece.loc[0], piece.loc[1]]
    return policy


def convert_moves_dict2vec(policy_dict: dict):
    list_policies = []
    for k, v in policy_dict.items():
        policy = []
        policy += convert_piece2vec(k.piece_moved)
        policy += k.destination
        policy += [v]
        # for pic in k.pieces_eaten:
        #     policy += convert_piece2vec(pic)
        list_policies += policy
    if len(list_policies) > POLICY_SIZE:
        print("len(list_policies)", len(list_policies))
    vec_size = len(list_policies)
    if vec_size < POLICY_SIZE:
        list_policies += [0] * (POLICY_SIZE - vec_size)
    else:
        list_policies = list_policies[:POLICY_SIZE]
    return list_policies
