from CheckersDisplay import CheckersDisplay
from Constants import WHITE, BLACK, PLAYER_NAME_A, PLAYER_NAME_B
from Game import Game
from PlayerFactory import PlayerFactory
from ReinforcementPlayer import ReinforcementPlayer


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
        while game_counter < self.num_of_games:
            # if self.game:
            self.game = Game(self.player1, self.player2, self.display)
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
            print(game_counter)
            if isinstance(self.player1, ReinforcementPlayer):
                self.player1.q_agent.decay_epsilon()
            if isinstance(self.player2, ReinforcementPlayer):
                self.player2.q_agent.decay_epsilon()
        print(f"BLACK: {self.black_score}, WHITE: {self.white_score}, Ties: {self.ties}")
        if self.display:
            self.display.show_end_result(self.white_score, self.black_score, self.ties)
        if isinstance(self.player1, ReinforcementPlayer):
            self.player1.save_object(PLAYER_NAME_A, self.num_of_games)
        if isinstance(self.player2, ReinforcementPlayer):
            self.player2.save_object(PLAYER_NAME_B, self.num_of_games)

    def run(self):
        if self.display:
            self.display.root.mainloop()
        else:
            self.run_game_loop()

    def set_num_of_games(self, new_num_of_games):
        self.num_of_games = new_num_of_games

    def reset_scores(self):
        self.white_score = 0
        self.black_score = 0
        self.ties = 0

if __name__ == "__main__":
    manager = GameManager(True)
    manager.run()


