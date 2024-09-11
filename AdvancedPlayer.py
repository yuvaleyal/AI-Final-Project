import os
from datetime import datetime

from Player import Player
from abcmeta import ABC, abstractmethod
from State import State
from Constants import *


class AdvancedPlayer(Player):

    def __init__(self, color: int) -> None:
        """constructor for the player class

        Args:
            color (int): BLACK or WHITE
        """
        super().__init__(color)
        self.color = color

    @abstractmethod
    def make_move(self, state: State) -> State:
        pass

    @abstractmethod
    def save_object(self):
        pass

    @abstractmethod
    def load_object(self):
        pass

    @abstractmethod
    def update_player(self, winner):
        pass

    @staticmethod
    def rename_old_state_file(file_path):
        if not os.path.exists(OBJECTS_DIR):
            os.makedirs(OBJECTS_DIR)
        if os.path.isfile(file_path):
            file_path_arr = file_path.split(".")
            file_path_arr[0] = f"{file_path_arr[0]}_{datetime.now().timestamp()}"
            file_path_new = ".".join(file_path_arr)
            os.rename(file_path, file_path_new)

    def clean_env(self):
        pass
