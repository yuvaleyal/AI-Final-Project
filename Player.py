from abcmeta import ABC, abstractmethod
from Constants import *
from State import State


class Player(ABC):
    def __init__(self, color: int) -> None:
        """constructor for the player class

        Args:
            color (int): BLACK or WHITE
        """
        self.color = color

    @abstractmethod
    def make_move(self, state: State) -> State:
        """makes a move on the board according to the specific algorithm implemented

        Args:
            state (State): the current state of the board

        Returns:
            State: the state of the board after the player's move
        """
        pass
