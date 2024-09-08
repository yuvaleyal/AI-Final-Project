import Player
from Constants import BLACK, WHITE
from HumanPlayer import HumanPlayer
from MinimaxPlayer import MinimaxPlayer
from RandomPlayer import RandomPlayer


class PlayerFactory:
    @staticmethod
    def get_player(player_type, player_num, display) -> Player:
        # define the color of the new player:
        color = WHITE if player_num == 1 else BLACK
        # Create the new player
        if player_type == 'human':
            return HumanPlayer(color, display)
        elif player_type == 'random':
            return RandomPlayer(color)
        elif player_type == 'minimax':
            return MinimaxPlayer(color)
        # elif player_type == 'rl':
        #     return ReinforcedLearninigPlayer(color)
        # elif player_type == 'neural_net':
        #     return NeuralNetworkPlayer(color)
        return None

