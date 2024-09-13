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
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0


def backup(node, value):
    while node:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        node = node.parent
        value = -value


class MCTS:
    def __init__(self, policy_value_fn, num_simulations, c_puct=1.0):
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def get_action(self, state):
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = root
            # Selection
            while node.children:
                node = self.select(node)
            # Expansion and Evaluation
            value = self.evaluate(node)
            # Backpropagation
            backup(node, value)
        # After simulations, select the action with highest visit count
        action_visits = [(move, child.N) for move, child in root.children.items()]
        move, _ = max(action_visits, key=lambda x: x[1])
        return move

    def get_policy_distribution(self, state):
        # Return the visit counts as probabilities
        root = MCTSNode(state)
        for _ in range(self.num_simulations):
            node = root
            while node.children:
                node = self.select(node)
            value = self.evaluate(node)
            backup(node, value)
        visit_counts = np.zeros(ACTION_SIZE)
        for move, child in root.children.items():
            idx = move_to_action_index(move)
            visit_counts[idx] = child.N
        policy = visit_counts / np.sum(visit_counts)
        return policy

    def select(self, node):
        total_N = sum(child.N for child in node.children.values())
        best_score = -float('inf')
        best_child = None
        for move, child in node.children.items():
            ucb = child.Q + self.c_puct * child.P * math.sqrt(total_N) / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

    def evaluate(self, node):
        board_tensor = state_to_tensor(node.state).unsqueeze(0).to(self.policy_value_fn.device)
        policy_logits, value = self.policy_value_fn(board_tensor)
        policy = torch.exp(policy_logits).detach().cpu().numpy()[0]
        value = value.item()
        legal_moves = node.state.find_all_moves()
        action_probs = {}
        for move in legal_moves:
            action_index = move_to_action_index(move)
            action_probs[move] = policy[action_index]

        sum_probs = sum(action_probs.values())
        if sum_probs > 0:
            for move in action_probs:
                action_probs[move] /= sum_probs
        else:
            num_actions = len(action_probs)
            for move in action_probs:
                action_probs[move] = 1.0 / num_actions
        node.children = {}
        for move, prob in action_probs.items():
            next_state = node.state.generate_successor(move)
            node.children[move] = MCTSNode(next_state, parent=node, prior=prob)
        return value
