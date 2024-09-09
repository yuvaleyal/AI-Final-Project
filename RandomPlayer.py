from Player import Player
from State import State
import random


class RandomPlayer(Player):
    """this player makes moves by random"""

    def __init__(self, color: int) -> None:
        super().__init__(color)

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        if not options:
            move = None
        else:
            move = random.choice(options)
        return state.next_state(move)
