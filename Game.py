from Board import Board
from CheckersDisplay import CheckersDisplay
from Constants import NOT_OVER_YET, BLACK, WHITE, CMD
from Pieces import RegularPiece
from Player import Player
from State import State


class Game:
    def __init__(self, player1: Player, player2: Player, display: CheckersDisplay):
        self.current_player: Player = player1
        self.player1 = player1
        self.player2 = player2
        self.display = display

        self.board = initialize_board()
        self.current_state = State(self.board, self.player2.color, [])

    def run(self):
        self.current_player = self.player1
        while self.current_state.is_over() == NOT_OVER_YET:
            self.current_state = self.current_player.make_move(self.current_state)
            if CMD[0]:
                print(self.current_state)
            if self.current_state.last_player == self.player1.color:
                self.current_player = self.player2
            else:
                self.current_player = self.player1
            if self.display:
                self.display.render_board()
        return self.current_state.is_over()


def initialize_board():
    black_locs = [(0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5), (1, 7), (2, 0), (2, 2), (2, 4), (2, 6)]
    black_pieces = [RegularPiece(BLACK, b_loc) for b_loc in black_locs]
    white_locs = [(5, 1), (5, 3), (5, 5), (5, 7), (6, 0), (6, 2), (6, 4), (6, 6), (7, 1), (7, 3), (7, 5), (7, 7)]
    white_pieces = [RegularPiece(WHITE, w_loc) for w_loc in white_locs]
    return Board(black_pieces, white_pieces)


# def init_board(self):
#     black_pieces = []
#     white_pieces = []
#
#     for row in range(3):
#         for col in range(4):
#             black_col = col * 2 + (row % 2)
#             black_pieces.append(RegularPiece(self.player2.color, (row, black_col)))
#
#             white_col = col * 2 + ((row + 1) % 2)
#             white_pieces.append(RegularPiece(self.player1.color, (BOARD_SIZE - 1 - row, white_col)))
#
#     return Board(black_pieces, white_pieces)
