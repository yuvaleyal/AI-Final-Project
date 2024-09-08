import Player
from Constants import BLACK, WHITE, PLAYER_NAME_A, PLAYER_NAME_B
from DNNPlayer import DNNPlayer
from FirstChoicePlayer import FirstChoicePlayer
from HumanPlayer import HumanPlayer
from MinimaxPlayer import MinimaxPlayer
from RandomPlayer import RandomPlayer
from ReinforcementPlayer import ReinforcementPlayer


class PlayerFactory:
    @staticmethod
    def get_player(player_type, player_num, display=None) -> Player:
        # define the color of the new player:
        color = player_num
        # Create the new player
        if player_type == 'random':
            return RandomPlayer(color)
        elif player_type == 'minimax':
            return MinimaxPlayer(color)
        elif player_type == 'rl':
            player = ReinforcementPlayer(color)
            if player_num == BLACK:
                player.load_object(PLAYER_NAME_A)
            elif player_num == WHITE:
                player.load_object(PLAYER_NAME_B)
            return player
        elif player_type == 'dnn':
            return DNNPlayer(color)
        elif player_type == 'first_choice':
            return FirstChoicePlayer(color)
        elif player_type == 'human':
            return HumanPlayer(color, display)
        return None

