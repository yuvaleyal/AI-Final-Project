from Player import Player
from State import State

class FirstChoicePlayer(Player):
    """this player makes moves by random"""

    def __init__(self, color: int) -> None:
        super().__init__(color)

    def make_move(self, state: State) -> State:
        options = state.find_all_moves()
        move = options[0]
        return state.next_state(move)
