import Player
from AlphaZeroPlayer import AlphaZeroPlayer
from Constants import AlphaZero, HUMAN, DQN
from DQNPlayer import DQN_Player
from FirstChoicePlayer import FirstChoicePlayer
from HumanPlayer import HumanPlayer
from MinimaxPlayer import MinimaxPlayer
from RandomPlayer import RandomPlayer
from ReinforcementPlayer import ReinforcementPlayer


class PlayerFactory:
    @staticmethod
    def get_player(player_type, player_num, display=None, train=False) -> Player:
        # define the color of the new player:
        color = player_num
        # Create the new player
        if player_type == 'random':
            return RandomPlayer(color)
        elif player_type == 'minimax':
            return MinimaxPlayer(color)
        elif player_type == 'rl':
            player = ReinforcementPlayer(color)
            player.load_object()
            return player
        elif player_type == AlphaZero:
            player = AlphaZeroPlayer(color)
            player.load_object()
            return player
        elif player_type == DQN:
            player = DQN_Player(color)
            player.load_object()
            return player
        elif player_type == 'first_choice':
            return FirstChoicePlayer(color)
        elif player_type == HUMAN:
            return HumanPlayer(color, display)
        return None

