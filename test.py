
from Board import Board
from Player import Player
from RandomPlayer import RandomPlayer
from State import State
from Constants import *
from Pieces import *





if __name__ == "__main__":
    # for black_pieces
    a1 = RegularPiece(BLACK, (6, 6))
    print(a1.immediate_move_options())
    a1 = RegularPiece(BLACK, (7, 7))
    print(a1.immediate_move_options())
    #
    b1 = RegularPiece(WHITE, (4, 0))
    print(b1.immediate_move_options())

    b1 = RegularPiece(WHITE, (1, 1))
    print(b1.immediate_move_options())
    b1 = RegularPiece(WHITE, (0, 0))
    print(b1.immediate_move_options())
