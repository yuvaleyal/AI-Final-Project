from Player import Player
from State import State
import random


class RandomPlayer(Player):
    """this player makes moves by random"""

    def __init__(self, color: int) -> None:
        super().__init__(color)

    def make_move(self, state: State):
        options = state.find_all_moves()
        move = random.choice(options)
        return move
        # if 7 < move.destination[0] < 0 or 0 < move.destination[1] >= 8:
        #     print("error")
        # return state.next_state(move)
