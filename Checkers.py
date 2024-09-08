from Board import Board
from DNNPlayer import DNNPlayer
from HumanPlayer import HumanPlayer
from MinimaxPlayer import MinimaxPlayer
from Player import Player
from RandomPlayer import RandomPlayer
from ReinforcementPlayer import ReinforcementPlayer
from State import State
from Constants import *
from Pieces import *
import argparse

TYPE_PLAYERS = ['random', 'human', 'minimax', 'rl', 'dnn']


def initialize_board():
    black_locs = [(0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5), (1, 7), (2, 0), (2, 2), (2, 4), (2, 6)]
    black_pieces = [RegularPiece(BLACK, b_loc) for b_loc in black_locs]
    white_locs = [(5, 1), (5, 3), (5, 5), (5, 7), (6, 0), (6, 2), (6, 4), (6, 6), (7, 1), (7, 3), (7, 5), (7, 7)]
    white_pieces = [RegularPiece(WHITE, w_loc) for w_loc in white_locs]
    return Board(black_pieces, white_pieces)


def main():
    """
    Processes the command used to run the game from the command line.
    """
    parser = argparse.ArgumentParser(description='Checkers game.')
    usage_str = """
    USAGE:      python Checkers.py <options>
    EXAMPLES:  (1) python Checkers.py
                  - starts a game between 2 random agents
               (2) python Checkers.py -a=random -b=random -d=cmd -n=3
    """

    parser.add_argument('-a', '--playerA',
                        choices=TYPE_PLAYERS,
                        help='Select player type A, the top player on the board, color white',
                        default='random')
    parser.add_argument('-b', '--playerB',
                        help='Select player type B, the bottom player on the board, color black', choices=TYPE_PLAYERS,
                        default='human')
    parser.add_argument('-d', '--display',
                        help='Run a display? yes, no, command line only = cmd.', default='cmd', type=str)
    parser.add_argument('-n', '--number_games',
                        help='The number of games that will be played one after the other in a row', default=1,
                        type=int)

    args = parser.parse_args()
    print("args:", args)
    if args.display == 'yes':
        pass
    elif args.display == 'no':
        pass
    elif args.display == 'cmd':
        pass

    if args.playerA in TYPE_PLAYERS:
        if args.playerA == 'random':
            playerA = RandomPlayer(BLACK)
        elif args.playerA == 'human':
            playerA = HumanPlayer(BLACK)
        elif args.playerA == 'minimax':
            playerA = MinimaxPlayer(BLACK)
        elif args.playerA == 'rl':
            playerA = ReinforcementPlayer(BLACK)
        elif args.playerA == 'dnn':
            playerA = DNNPlayer(BLACK)

    if args.playerB in TYPE_PLAYERS:
        if args.playerB == 'random':
            playerB = RandomPlayer(BLACK)
        elif args.playerB == 'human':
            playerB = HumanPlayer(BLACK)
        elif args.playerB == 'minimax':
            playerB = MinimaxPlayer(BLACK)
        elif args.playerB == 'rl':
            playerB = ReinforcementPlayer(BLACK)
        elif args.playerA == 'dnn':
            playerB = DNNPlayer(BLACK)

    else:
        raise Exception('unrecognized options')

    board = initialize_board()

    current_player = playerA
    state = State(board, BLACK)
    n_games = args.number_games
    while n_games > 0:
        while state.is_over() == NOT_OVER_YET:
            print(state.get_board_list())  # Print the current board state
            print(state)
            print(f"{current_player.color}'s turn")
            print(state.find_all_moves())
            state = current_player.make_move(state)
            if state.last_player == BLACK:
                current_player = playerA
            elif state.last_player == WHITE:
                current_player = playerB
            else:
                print("Invalid move, try again.")

        # Announce the winner
        if (final := state.is_over()) != NOT_OVER_YET:
            if final == BLACK:
                print("Black wins!")
            elif final == WHITE:
                print("White wins!")
            elif final == TIE:
                print("Tie!")
        n_games -= 1


if __name__ == "__main__":
    main()
