from Player import Player
from State import State


class ReinforcementPlayer(Player):
    def __init__(self, color: int) -> None:
        super().__init__(color)

    def make_move(self, state: State) -> State:
        pass

