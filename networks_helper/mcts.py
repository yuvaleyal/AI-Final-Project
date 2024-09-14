# networks_helper/mcts.py
import math
import numpy as np
import torch
from Constants import *
from networks_helper.func_helper import state_to_tensor, move_to_action_index


class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.P = prior  # Prior probability (from the policy network)
        self.N = 0  # Visit count
        self.W = 0.0  # Total action value
        self.Q = 0.0  # Mean action value

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self, action_probs):
        for move, prob in action_probs.items():
            if move not in self.children:
                next_state = self.state.generate_successor(move)
                self.children[move] = MCTSNode(next_state, parent=self, prior=prob)

    def update(self, leaf_value):
        self.N += 1
        self.W += leaf_value
        self.Q = self.W / self.N

    def update_recursive(self, leaf_value):
        # Update until root node
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        u = (c_puct * self.P * math.sqrt(self.parent.N) / (1 + self.N))
        return self.Q + u


class MCTS:
    def __init__(self, policy_value_fn, num_simulations, c_puct=1.0, device='cpu'):
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.device = device

    def playout(self, state):
        node = self.root
        # Selection
        while True:
            if node.is_leaf():
                break
            # Select next move
            max_value = -np.inf
            best_move = None
            for move, child in node.children.items():
                value = child.get_value(self.c_puct)
                if value > max_value:
                    max_value = value
                    best_move = move
            node = node.children[best_move]
        # Evaluation
        board_tensor = state_to_tensor(node.state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.policy_value_fn(board_tensor)
        policy = torch.exp(policy_logits).cpu().numpy()[0]
        value = value.item()
        legal_moves = node.state.find_all_moves()
        if not legal_moves:
            # Game over, current player loses
            leaf_value = -1.0  # Current player loses
            node.update_recursive(-leaf_value)
            return

        action_probs = {}
        for move in legal_moves:
            action_index = move_to_action_index(move)
            action_probs[move] = policy[action_index]

        # Normalize the probabilities
        sum_probs = sum(action_probs.values())
        if sum_probs > 0:
            for move in action_probs:
                action_probs[move] /= sum_probs
        else:
            # If all probs are zero, set them uniformly
            prob = 1.0 / len(action_probs)
            for move in action_probs:
                action_probs[move] = prob

        # Expansion
        node.expand(action_probs)
        # Backpropagation
        node.update_recursive(-value)

    def get_action(self, state, temp=1e-3):
        self.root = MCTSNode(state)
        for _ in range(self.num_simulations):
            self.playout(state)

        # Get the move probabilities
        acts = []
        visits = []
        for move, child in self.root.children.items():
            acts.append(move)
            visits.append(child.N)
        acts_visits = list(zip(acts, visits))
        acts_visits.sort(key=lambda x: x[1], reverse=True)
        moves, counts = zip(*acts_visits)

        counts = np.array(counts, dtype=np.float64)

        if temp <= 1e-3:
            # When temp is very low, select the move with the highest visit count
            probs = np.zeros(len(counts))
            best_move_idx = np.argmax(counts)
            probs[best_move_idx] = 1.0
        else:
            counts_sum = np.sum(counts)
            if counts_sum == 0:
                probs = np.ones(len(counts)) / len(counts)
            else:
                # Apply temperature to the counts using a softmax function
                counts = counts / counts_sum  # Normalize counts to sum to 1
                counts = np.log(counts + 1e-10) / temp  # Avoid log(0) by adding a small constant
                counts = counts - np.max(counts)  # For numerical stability
                exp_counts = np.exp(counts)
                probs = exp_counts / np.sum(exp_counts)

        # Select move
        move = np.random.choice(moves, p=probs)

        # Prepare the policy vector
        pi = np.zeros(ACTION_SIZE)
        for move_i, prob in zip(moves, probs):
            idx = move_to_action_index(move_i)
            pi[idx] = prob

        return move, torch.tensor(pi, dtype=torch.float32)
