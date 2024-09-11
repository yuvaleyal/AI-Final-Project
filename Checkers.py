from Board import Board
from CheckersGameManager import GameManager
from DNNPlayer import DNNPlayer
from FirstChoicePlayer import FirstChoicePlayer
from HumanPlayer import HumanPlayer
from MinimaxPlayer import MinimaxPlayer
from Player import Player
from RandomPlayer import RandomPlayer
from ReinforcementPlayer import ReinforcementPlayer
from State import State
from Constants import *
from Pieces import *
import argparse


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
    parser.add_argument('-m', '--mode',
                        choices=PROG_MODE,
                        help=f'Select mode of game. training, evaluation or test mode {PROG_MODE}',
                        default='eval')
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

    MODE[0] = args.mode
    print("Game Mode:", MODE)
    if args.display == 'yes':
        game_manager = GameManager(args.display)
        game_manager.run()
    elif args.display in ['no', 'cmd']:
        if args.display == 'cmd':
            CMD[0] = True
        game_manager = GameManager()
        if args.playerA == HUMAN or args.playerB == HUMAN:
            raise Exception('The human player is only possible in display mode')

        if args.playerA in TYPE_PLAYERS and args.playerB in TYPE_PLAYERS:
            game_manager.init_game(args.playerA, args.playerB)
        game_manager.set_num_of_games(args.number_games)
        game_manager.run()


if __name__ == "__main__":
    main()
