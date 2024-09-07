from abcmeta import ABC, abstractmethod
from Constants import *
# from Board import Board


class Piece(ABC):
    def __init__(self, player, loc, board_size=BOARD_SIZE):
        """constructor for the piece class

        Args:
            player (int): BLACK or WHITE. the valie corresponds to the direction the piece movges on the board
            loc (tuple: (int, int)): the piece location on the board, as (row, col)
            board_size (int, optional): the size of the board the pieces are on. Defaults to BOARD_SIZE.
        """
        self.player = player
        self.loc = loc
        self.board_size = board_size

    def get_player(self):
        """returns the player the piece belongs to

        Returns:
            int: BLACK or WHITE
        """
        return self.player

    def get_location(self):
        """returns the piece loacation on the board

        Returns:
            tuple: (int, int): the piece location on the board, as (row, col)
        """
        return self.loc

    def set_location(self, location: tuple[int, int]):
        """sets the piece location

        Args:
            location (tuple[int, int]): the new location
        """
        self.loc = location

    @abstractmethod
    def immediate_move_options(self) -> list[tuple[int, int]]:
        """return a list of the locations the piece can move to (if he board was otherwise empty)"""
        pass

    @abstractmethod
    def is_queen(self) -> bool:
        """
        Returns:
            bool: is the piece a queen
        """
        pass

    def __repr__(self):
        return f"Piece({self.player}, loc:({self.loc}))"


class RegularPiece(Piece):
    def __init__(self, player, loc):
        super().__init__(player, loc)

    def immediate_move_options(self) -> list[tuple[int, int]]:
        options = []
        if 0 < self.loc[0] < self.board_size - 1:
            if self.loc[1] != 0:
                options.append((self.loc[0] + self.player, self.loc[1] - 1))
            if self.loc[1] != self.board_size - 1:
                options.append((self.loc[0] + self.player, self.loc[1] + 1))
        return options

    def is_queen(self) -> bool:
        return False


"""    def get_all_moves(self, board: Board):
        paths = []
        while True:
            options = self.immediate_move_options()
            for option in options:
                piece = board.get_piece(option)
                if piece == None:
                    paths += [[option]]
                if piece.get_player != self.player:
                    #is next empty? if so can chain?
                    """


class QueenPiece(Piece):
    def __init__(self, player, loc):
        super().__init__(player, loc)

    def immediate_move_options(self) -> list[tuple[int, int]]:
        options = []
        for i, j in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            row, col = self.loc
            while 0 <= row + i < self.board_size and 0 <= col + j < BOARD_SIZE:
                row += i
                col += j
                options.append((row, col))
        return options

    def is_queen(self) -> bool:
        return True
