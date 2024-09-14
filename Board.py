import numpy as np
from Constants import *
from Pieces import Piece, RegularPiece, QueenPiece
from Move import Move


class Board:
    def __init__(self, black_pieces: list[Piece], white_pieces: list[Piece]):
        self.black_pieces = black_pieces
        self.white_pieces = white_pieces

    def get_piece(self, loc: tuple[int, int]) -> Piece:
        """returns the piece on the given location, None if its empty
        assumes the location is on the board and that there are no 2 pieces in the same location

        Args:
            loc (tuple[int, int]): the given location

        Returns:
            Piece: the piece if exists, None if the spot is empty
        """
        for piece in self.black_pieces:
            if piece.get_location() == loc:
                return piece
        for piece in self.white_pieces:
            if piece.get_location() == loc:
                return piece
        return None

    def get_pieces(self, player: int) -> list[Piece]:
        """returns all the pieces of the player

        Args:
            player (int): BLACK or WHITE

        Returns:
            list[Piece]: the list of pieces
        """
        if player == BLACK:
            return self.black_pieces
        if player == WHITE:
            return self.white_pieces
        return None

    def make_move(self, move: Move) -> None:
        """makes a move on the board - moves a piece between locations and remove the eaten pieces from the board.
        assumes the move is legal and all the required pieces are on the board

        Args:
            move (Move): the move to be made
        """
        ### to check for the weird problem:
        cur_black_num, cur_white_num = len(self.black_pieces), len(self.white_pieces)
        ###
        piece_moved = move.get_piece_moved()
        piece_moved.set_location(move.get_destination())
        if move.get_destination()[0] == self._get_other_end(piece_moved.get_player()):
            if not piece_moved.is_queen():
                self.get_pieces(piece_moved.get_player()).remove(piece_moved)
                self.get_pieces(piece_moved.get_player()).append(
                    QueenPiece(piece_moved.get_player(), piece_moved.get_location()))
        for piece in move.get_pieces_eaten():
            self._remove_piece(piece)

        ##checking for the weird problem:
        if len(self.black_pieces) > cur_black_num or len(self.white_pieces) > cur_white_num:
            #so, somehow pieces where added...
            #place breakpoint here:
            pass

    # for now, nothing
    def can_eat(self, piece: Piece):
        cur_player = piece.get_player()
        cur_loc = piece.get_location()
        if piece not in self.get_pieces(cur_player):
            return
        options = []
        if not piece.is_queen():
            for loc in piece.move_options():
                target = self.get_piece(loc)
                if (
                    target is not None
                    and target.get_player() != cur_player
                    and self.get_piece(self.next_step(cur_loc, loc)) is None
                ):
                    options += [target]
            return options
        # a queen

        for loc in piece.move_options():
            if self.get_piece(loc).get_player() != cur_player:
                path, next = self._path_on_board(piece.get_location(), loc)
                if len(path) == 0:  # immediate next step
                    pass

    # private methods:

    def _is_on_board(self, piece: Piece):
        if piece.get_player() == WHITE:
            return piece in self.white_pieces
        return piece in self.black_pieces

    def _remove_piece(self, piece: Piece):
        if piece.get_player() == WHITE:
            self.white_pieces.remove(piece)
        else:
            self.black_pieces.remove(piece)

    def _get_other_end(self, player: int) -> int:
        if player == BLACK:
            return BOARD_SIZE - 1
        return 0
