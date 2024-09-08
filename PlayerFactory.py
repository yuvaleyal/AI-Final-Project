import Player
from Constants import BLACK, WHITE
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
            return ReinforcementPlayer(color)
        elif player_type == 'dnn':
            return DNNPlayer(color)
        elif player_type == 'first_choice':
            return FirstChoicePlayer(color)
        elif player_type == 'human':
            return HumanPlayer(color, display)
        return None

