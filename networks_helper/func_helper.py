from Constants import *


def state_to_tensor(state):
    import torch
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
    if (row + col) % 2 == 1:
        print("ERROR, square_to_index", (row + col) % 2 == 1)
    # print(f"test{(row, col)},to {(row * 4) + (col // 2)}, back {index_to_square((row * 4) + (col // 2))}")
    #
    if (row + col) % 2 == 0:
        return (row * 4) + (col // 2)
    else:
        return None


def index_to_square(index):
    row = index // 4
    col = (index % 4) * 2
    if (row + col) % 2 == 0:
        col += 0
    else:
        col += 1
    return row, col
