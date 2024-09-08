from CheckersDisplay import CheckersDisplay
from Constants import WHITE, BLACK
from Game import Game
from PlayerFactory import PlayerFactory


class GameManager:
    def __init__(self):
        # Initialize the display
        self.num_of_games = 0
        self.display_board = None
        self.game = None
        self.display = CheckersDisplay(self)
        self.black_score = 0
        self.white_score = 0
        self.ties = 0

    def init_game(self, player1_type, player2_type):
        # Initialize the game with the selected player types
        player1 = PlayerFactory.get_player(player1_type, 1, self.display)
        player2 = PlayerFactory.get_player(player2_type, 2, self.display)
        self.game = Game(player1, player2, self.display, self.display_board)

    def run_game_loop(self):
        # Start the game loop
        game_counter = 0
        while game_counter < self.num_of_games:
            if self.game:
                winner = self.game.run()
                if winner == WHITE:
                    self.white_score += 1
                elif winner == BLACK:
                    self.black_score += 1
                else:
                    self.ties += 1
                game_counter += 1
                if not self.display_board:
                    self.display.update_progress_bar(game_counter)
            else:
                break
        self.display.show_end_result(self.white_score, self.black_score, self.ties)

    def run(self):
        self.display.root.mainloop()

    def set_num_of_games(self, new_num_of_games):
        self.num_of_games = new_num_of_games

    def reset_scores(self):
        self.white_score = 0
        self.black_score = 0
        self.ties = 0

    def set_board_display(self, should_display_board):
        self.display_board = should_display_board

if __name__ == '__main__':
    # Initialize GameManager and start the Dash server
    game_manager = GameManager()
    game_manager.run()
