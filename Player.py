from abcmeta import ABC, abstractmethod
from Constants import *
from State import State


class Player(ABC):
    def __init__(self, color: int) -> None:
        self.color = color

    @abstractmethod
    def make_move(self, state: State) -> State:
        pass
