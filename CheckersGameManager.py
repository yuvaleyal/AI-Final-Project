from AdvancedPlayer import AdvancedPlayer
from AlphaZeroPlayer import AlphaZeroPlayer
from CheckersDisplay import CheckersDisplay
from Constants import *
from Game import Game
from PlayerFactory import PlayerFactory


class GameManager:
    def __init__(self, display=None):
        # Initialize the display
        self.num_of_games = 0
        self.display_board = None
        self.game = None
        if display:
            self.display = CheckersDisplay(self)
        else:
            self.display = None
        self.black_score = 0
        self.white_score = 0
        self.ties = 0
        self.player1 = None
        self.player2 = None

    def init_game(self, player1_type, player2_type):
        # Initialize the game with the selected player types
        self.player1 = PlayerFactory.get_player(player1_type, BLACK, self.display)
        self.player2 = PlayerFactory.get_player(player2_type, WHITE, self.display)

    def run_game_loop(self):
        # Start the game loop
        game_counter = 0
        if MODE[0] != TRAINING_MODE:
            self.player1.rl_update = False

        while game_counter < self.num_of_games:
            if game_counter % 2 == 0:
                self.game = Game(self.player1, self.player2, self.display)
            else:
                self.game = Game(self.player2, self.player1, self.display)
            if self.display:
                self.display.render_board()
            winner = self.game.run()
            if winner == WHITE:
                self.white_score += 1
            elif winner == BLACK:
                self.black_score += 1
            else:
                self.ties += 1
            game_counter += 1
            if CMD[0]:
                print(game_counter)
            if MODE[0] == TRAINING_MODE or MODE[0] == TESTING_MODE:
                self.update_loop_game(game_counter, winner)
            if self.display:
                self.display.update_scores(self.black_score, self.white_score)
        print(f"BLACK: {self.black_score}, WHITE: {self.white_score}, Ties: {self.ties}")
        if self.display:
            self.display.show_end_result(self.black_score, self.white_score, self.ties)
        if MODE[0] == TRAINING_MODE:
            self.save_game()

    def run(self):
        if self.display:
            self.display.root.mainloop()
        else:
            self.run_game_loop()

    def update_loop_game(self, game_counter, winner):
        print(
            f"Episode {game_counter}/{self.num_of_games},"
            f" BLACK: {self.black_score}, WHITE: {self.white_score}, Ties: {self.ties}")
        if MODE[0] == TRAINING_MODE:
            if isinstance(self.player1, AdvancedPlayer):
                self.player1.update_player(winner)
                if isinstance(self.player1, AlphaZeroPlayer) and game_counter % 10 == 0:
                    self.player1.save_object()
            if isinstance(self.player2, AdvancedPlayer):
                self.player2.update_player(winner)

    def save_game(self):
        if isinstance(self.player1, AdvancedPlayer):
            self.player1.save_object()
        if isinstance(self.player2, AdvancedPlayer):
            self.player2.save_object()

    def set_num_of_games(self, new_num_of_games):
        self.num_of_games = new_num_of_games

    def reset_scores(self):
        self.white_score = 0
        self.black_score = 0
        self.ties = 0
