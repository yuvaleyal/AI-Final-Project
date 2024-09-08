from Board import Board
from Move import Move
from Pieces import Piece, RegularPiece, QueenPiece
import numpy as np
from Constants import *


class State:
    def __init__(self, cur_state: Board, last_player: int) -> None:
        self.board = cur_state
        self.last_player = last_player
        self.next_moves = []

    def find_all_moves(self) -> list[Move]:
        """returns a list of all the possible moves for the next player
        FOR NOW: DOESNT INCLUDE CHAIN MOVES

        Returns:
            list[Move]: all the possible moves
        """
        moves = []
        can_eat = False
        for piece in self.board.get_pieces(-self.last_player):
            loc = piece.get_location()
            for option in piece.immediate_move_options():
                steps = self._path_to_location(loc, option)
                if all(self.board.get_piece(step) is None for step in steps):
                    piece_in_dest = self.board.get_piece(option)
                    if piece_in_dest is None:
                        moves.append(
                            Move(piece_moved=piece, destination=option, pieces_eaten=[])
                        )
                    elif piece_in_dest.get_player() == self.last_player:
                        after_eating_option_loc = self.next_step(loc, option)
                        if self.board.get_piece(after_eating_option_loc) is None and self._loc_in_board(
                            after_eating_option_loc):
                            eat_one_move = Move(
                                    piece_moved=piece,
                                    destination=after_eating_option_loc,
                                    pieces_eaten=[piece_in_dest],
                                )
                            moves.append(eat_one_move)
                            moves += self.make_chain(piece, eat_one_move)
                            can_eat = True
        if can_eat:
            return [move for move in moves if len(move.get_pieces_eaten()) > 0]
        return moves

    def next_state(self, move: Move):
        """returns the state after the current player makes a move

        Args:
            move (Move): the move to be made

        Returns:
            State: next state
        """
        self.board.make_move(move)
        return State(self.board, -self.last_player)

    def is_over(self) -> int:
        """checks the board state and returns whether the game is not over, won by one of the players or a tie.
        a tie is defined by each of the players having a single queen and nothing else

        Returns:
            int: BLACK, WHITE, TIE or NOT_OVER_YET
        """
        black_pieces = self.board.get_pieces(BLACK)
        if len(black_pieces) == 0:
            return WHITE
        white_pieces = self.board.get_pieces(WHITE)
        if len(white_pieces) == 0:
            return BLACK
        if (
            len(black_pieces) == 1
            and len(white_pieces) == 1
            and black_pieces[0].is_queen()
            and white_pieces[0].is_queen()
        ):
            return TIE
        return NOT_OVER_YET

    def get_board_list(self) -> list[list[int]]:
        """returns a list of lists representing the board

        Returns:
            list[list[int]]: representaion of the current state of the board
        """
        board_list = [[0] * BOARD_SIZE for i in range(BOARD_SIZE)]
        for player in [BLACK, WHITE]:
            l = self.board.get_pieces(player)
            for piece in self.board.get_pieces(player):
                row, col = piece.get_location()
                if piece.is_queen():
                    board_list[row][col] = player * QUEEN_MULTIPLIER
                else:
                    board_list[row][col] = player
        return board_list

    # private methods:
    def _path_to_location(
        self, loc: tuple[int, int], dest: tuple[int, int]
    ) -> list[tuple[int, int]]:
        start_row, start_col = loc
        end_row, end_col = dest
        diff = end_row - start_row
        if abs(end_col - start_col) != abs(diff):
            return
        return [
            (
                start_row + i * np.sign(diff),
                start_col + i * np.sign(end_col - start_col),
            )
            for i in range(1, abs(diff))
        ]

    def next_step(self, loc: tuple[int, int], dest: tuple[int, int]):
        cur_row, cur_col = loc
        tar_row, tar_col = dest
        return (
            tar_row + np.sign(tar_row - cur_row),
            tar_col + np.sign(tar_col - cur_col),
        )

    def __repr__(self):
        show_board = self.get_board_list()
        str_ = ""
        for ind_row in range(len(show_board) -1, -1, -1):
            row = show_board[ind_row]
            str_ += ' '.join(f'{num:3}' for num in row)
            str_ += "\n"
        return str_

    def _loc_in_board(self, loc: tuple[int, int]) -> bool:
        return 0 <= loc[0] < BOARD_SIZE and 0 <= loc[1] < BOARD_SIZE

    def make_chain(self, piece: Piece, eat_move: Move) -> list[Move]:
        chain_options = []
        queue = [eat_move]
        while len(queue) > 0:
            cur_move = queue.pop(0)
            if piece.is_queen():
                temp_piece = QueenPiece(piece.get_player(), cur_move.get_destination())
            else:
                temp_piece = RegularPiece(piece.get_player(), cur_move.get_destination())
            for option in temp_piece.immediate_move_options():
                piece_in_dest = self.board.get_piece(option)
                if piece_in_dest is not None and piece_in_dest.get_player() != temp_piece.get_player() and piece_in_dest not in cur_move.get_pieces_eaten():
                    dest = self.next_step(cur_move.get_destination(), option)
                    if self.board.get_piece(dest) is None:
                        new_move = Move(piece, dest, cur_move.get_pieces_eaten() + [piece_in_dest])
                        chain_options.append(new_move)
                        queue.append(new_move)
        return chain_options
                