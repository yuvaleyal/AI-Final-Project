# networks_helper/func_helper.py
import torch
from Constants import *
import numpy as np

from State import State


def state_to_tensor(state:State):
    board_tensor = torch.zeros((4, 8, 8), dtype=torch.float32)
    current_player = -state.last_player
    board_list = state.get_board_list()
    for row in range(8):
        for col in range(8):
            piece_val = board_list[row][col]
            if piece_val == 0:
                continue
            if piece_val == current_player:
                board_tensor[0][row][col] = 1.0
            elif piece_val == current_player * QUEEN_MULTIPLIER:
                board_tensor[1][row][col] = 1.0
            elif piece_val == -current_player:
                board_tensor[2][row][col] = 1.0
            elif piece_val == -current_player * QUEEN_MULTIPLIER:
                board_tensor[3][row][col] = 1.0
    return board_tensor


def move_to_action_index(move):
    from_row, from_col = move.get_piece_moved().get_location()
    to_row, to_col = move.get_destination()
    from_index = square_to_index(from_row, from_col)
    to_index = square_to_index(to_row, to_col)
    if from_index is None or to_index is None:
        raise ValueError(f"Invalid square index for move from {from_row, from_col} to {to_row, to_col}")
    action_index = from_index * 32 + to_index
    return action_index


def square_to_index(row, col):
    if (row + col) % 2 != 0:
        return None
    return (row * 4) + (col // 2)


def index_to_square(index):
    row = index // 4
    col = (index % 4) * 2
    if (row + col) % 2 != 0:
        col += 1
    return row, col


def augment_data(states, mcts_probs, winners):
    augmented_states = []
    augmented_probs = []
    augmented_winners = []
    for i in range(len(states)):
        state = states[i]
        prob = mcts_probs[i]
        winner = winners[i]
        # Original
        augmented_states.append(state)
        augmented_probs.append(prob)
        augmented_winners.append(winner)
        # Flipped horizontally
        flipped_state = state.flip(-1)
        flipped_prob = flip_policy(prob, flip_horizontal=True)
        augmented_states.append(flipped_state)
        augmented_probs.append(flipped_prob)
        augmented_winners.append(winner)
        # Flipped vertically
        flipped_state = state.flip(-2)
        flipped_prob = flip_policy(prob, flip_vertical=True)
        augmented_states.append(flipped_state)
        augmented_probs.append(flipped_prob)
        augmented_winners.append(winner)
    return augmented_states, augmented_probs, augmented_winners


def flip_policy(policy, flip_horizontal=False, flip_vertical=False):
    # Reshape policy to 32x32 matrix
    policy = policy.view(32, 32)
    if flip_horizontal:
        policy = policy.flip(1)
    if flip_vertical:
        policy = policy.flip(0)
    # Flatten back to vector
    policy = policy.reshape(-1)
    return policy
