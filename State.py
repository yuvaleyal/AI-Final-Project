from Board import Board
from Move import Move
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
                        if self.board.get_piece(after_eating_option_loc) is None:
                            moves.append(
                                Move(
                                    piece_moved=piece,
                                    destination=after_eating_option_loc,
                                    pieces_eaten=[piece_in_dest],
                                )
                            )
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
