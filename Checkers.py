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


# def initialize_board():
#     black_locs = [(0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5), (1, 7), (2, 0), (2, 2), (2, 4), (2, 6)]
#     black_pieces = [RegularPiece(BLACK, b_loc) for b_loc in black_locs]
#     white_locs = [(5, 1), (5, 3), (5, 5), (5, 7), (6, 0), (6, 2), (6, 4), (6, 6), (7, 1), (7, 3), (7, 5), (7, 7)]
#     white_pieces = [RegularPiece(WHITE, w_loc) for w_loc in white_locs]
#     return Board(black_pieces, white_pieces)


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
    global CMD

    # if CMD:
    print("args:", args)
    if args.display == 'yes':
        game_manager = GameManager(args.display)
        game_manager.run()
    elif args.display == 'no':
        game_manager = GameManager()
        #     pass
        # elif args.display == 'cmd':
        #     CMD = True
        if args.playerA in TYPE_PLAYERS and args.playerB in TYPE_PLAYERS:
            game_manager.init_game(args.playerA, args.playerB)

        game_manager.set_num_of_games(args.number_games)
        game_manager.run()
    #     if args.playerA == 'random':
    #         playerA = RandomPlayer(BLACK)
    #     elif args.playerA == 'human':
    #         playerA = HumanPlayer(BLACK)
    #     elif args.playerA == 'minimax':
    #         playerA = MinimaxPlayer(BLACK)
    #     elif args.playerA == 'rl':
    #         playerA = ReinforcementPlayer(BLACK)
    #         playerA.load_object(PLAYER_NAME_A)
    #     elif args.playerA == 'dnn':
    #         playerA = DNNPlayer(BLACK)
    #     elif args.playerA == 'first_choice':
    #         playerA = FirstChoicePlayer(BLACK)
    #
    # if args.playerB in TYPE_PLAYERS:
    #     if args.playerB == 'random':
    #         playerB = RandomPlayer(WHITE)
    #     elif args.playerB == 'human':
    #         playerB = HumanPlayer(WHITE)
    #     elif args.playerB == 'minimax':
    #         playerB = MinimaxPlayer(WHITE)
    #     elif args.playerB == 'rl':
    #         playerB = ReinforcementPlayer(WHITE)
    #         playerB.load_object(PLAYER_NAME_B)
    #     elif args.playerA == 'dnn':
    #         playerB = DNNPlayer(WHITE)
    #     elif args.playerB == 'first_choice':
    #         playerB = FirstChoicePlayer(WHITE)
    else:
        raise Exception('unrecognized options')

    # wins_dict = {BLACK: 0, WHITE: 0, TIE: 0}
    # the_game_winner = ""
    # while n_games > 0:
    #     board = initialize_board()
    #     current_player = playerA
    #     state = State(board, WHITE)
    #     # print("q_table:", playerA.g_agent.q_table)
    #     # if CMD:
    #     #     print("q_table num_games:", playerA.q_agent.num_games)
    #     while state.is_over() == NOT_OVER_YET:
    #         if CMD:
    #             print(state)  # Print the current board state
    #             print(f"{current_player.color}'s turn")
    #             print(state.find_all_moves())
    #         state = current_player.make_move(state)
    #         if state.last_player == BLACK:
    #             current_player = playerA
    #         elif state.last_player == WHITE:
    #             current_player = playerB
    #         else:
    #             print("Invalid move, try again.")
    #
    #     # Announce the winner
    #     if (final := state.is_over()) != NOT_OVER_YET:
    #         if final == BLACK:
    #             the_game_winner = "Black wins!"
    #             wins_dict[BLACK] += 1
    #         elif final == WHITE:
    #             the_game_winner = "White wins!"
    #             wins_dict[WHITE] += 1
    #         elif final == TIE:
    #             the_game_winner = "Tie!"
    #             wins_dict[TIE] += 1
    #     if CMD:
    #         print(state)
    #         print(the_game_winner)
    #
    #     n_games -= 1
    #     if args.playerA == 'rl':
    #         playerA.q_agent.decay_epsilon()
    #     if args.playerB == 'rl':
    #         playerB.q_agent.decay_epsilon()
    #
    # if args.playerA == 'rl':
    #     playerA.save_object(PLAYER_NAME_A, args.number_games)
    # if args.playerB == 'rl':
    #     playerB.save_object(PLAYER_NAME_B, args.number_games)
    # print(wins_dict)


if __name__ == "__main__":
    main()
