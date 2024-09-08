import copy
import threading
import time
from Board import Board
from Constants import BOARD_SIZE, NOT_OVER_YET, BLACK
from Move import Move
from Pieces import Piece, RegularPiece
from Player import Player
from State import State
from CheckersDisplay import CheckersDisplay
from dash import Output


class Game:
    def __init__(self, player1: Player, player2: Player, display: CheckersDisplay, should_display_board: bool):
        self.current_player: Player = player1
        self.player1 = player1
        self.player2 = player2
        self.display = display
        self.should_display_board = should_display_board

        self.board = self.init_board()
        self.current_state = State(self.board, self.player2.color)

    def init_board(self):
        black_pieces = []
        white_pieces = []

        for row in range(3):
            for col in range(4):
                black_col = col * 2 + (row % 2)
                black_pieces.append(RegularPiece(self.player2.color, (row, black_col)))

                white_col = col * 2 + ((row + 1) % 2)
                white_pieces.append(RegularPiece(self.player1.color, (BOARD_SIZE - 1 - row, white_col)))

        return Board(black_pieces, white_pieces)

    # def update_board(self):
    #     self.current_player.make_move(self.current_state)
    #     self.display.update_board(self)  # Update the display with new board state

    def run(self):
        while self.current_state.is_over() == NOT_OVER_YET:
            move = self.current_player.make_move(self.current_state)
            if move is None:
                continue
            move_start = move.get_piece_moved().get_location()
            self.current_state = self.current_state.next_state(move)
            # self.current_state = State(self.board, self.current_player.color)
            self.current_player = self.player1 if self.current_state.last_player == BLACK else self.player2
            if self.should_display_board:
                self.display.update_board(move_start,
                                          move.get_destination(),
                                          move.get_pieces_eaten(),
                                          move.get_piece_moved().get_player())
            # time.sleep(3)
        return self.current_state.is_over()
