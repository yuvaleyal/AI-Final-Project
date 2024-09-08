# from Player import Player
# from State import State
#
#
# class HumanPlayer(Player):
#
#     def __init__(self, color: int) -> None:
#         super().__init__(color)
#
#     def make_move(self, state: State) -> State:
#         pass
from CheckersDisplay import CheckersDisplay
from Constants import BOARD_SIZE, WHITE
# from Pieces import EatingPiece
from Player import Player
from Move import Move
from Board import Board
from State import State
import tkinter as tk


class HumanPlayer(Player):
    def __init__(self, color: int, display: CheckersDisplay):
        super().__init__(color)
        self.legal_piece_moves = None
        self.display = display
        self.selected_piece = None
        self.legal_moves = []
        self.previous_piece = None
        self.move = None  # Variable to store the move

    def make_move(self, state: State) -> State:
        """Sets up the player's move process by binding the click events."""
        self.state = state
        self.selected_piece = None
        self.legal_moves = state.find_all_moves()
        self.move = None

        # If we are during a continuous eating move, continue to highlight the eating piece.
        self.highlight_eating_piece()

        # Bind the click event to handle the move
        self.display.canvas.bind("<Button-1>", self.on_click)
        while self.move is None:
            self.display.canvas.update()

        # Wait for the move to be set
        self.check_for_move()  # This keeps the UI responsive while waiting

        self.display.canvas.unbind("<Button-1>")
        return self.move

    def check_for_move(self):
        """Periodically checks if a move has been made."""
        if self.move is not None:
            # Unbind the click event since the move is complete
            self.display.canvas.unbind("<Button-1>")
        else:
            # Continue waiting by scheduling another check
            self.display.canvas.after(100, self.check_for_move)

    def on_click(self, event):
        """Handle mouse click events to select pieces and make moves."""
        row = int(event.y // self.display.cell_size)
        col = int(event.x // self.display.cell_size)
        print(f"Clicked at: row={row}, col={col}")

        piece = self.state.board.get_piece((row, col))

        if self.selected_piece:
            move = self.get_move_to_location(row, col)
            if move:
                self.move = move  # Set the move and exit the loop
                self.display.clear_highlights()
                print(f"Move selected from {self.selected_piece.get_location()} to {row}, {col}")
            else:
                print("Invalid move. Select another valid destination.")

        # If a valid piece is clicked, update the selected piece
        if piece and piece.get_player() == self.color and self.is_movable_piece(piece):
            self.selected_piece = piece
            self.previous_piece = self.selected_piece
            self.legal_piece_moves = self.state.get_piece_moves(piece)
            self.display.highlight_legal_moves(self.selected_piece, self.legal_piece_moves)
            print(f"Selected piece at {row}, {col}")

        # # If a piece is already selected, check if the clicked cell is a legal move
        # elif self.selected_piece:
        #     move = self.get_move_to_location(row, col)
        #     if move:
        #         self.move = move  # Set the move and exit the loop
        #         print(f"Move selected from {self.selected_piece.get_location()} to {row}, {col}")
        #     else:
        #         print("Invalid move. Select another valid destination.")

    def is_movable_piece(self, piece):
        for move in self.legal_moves:
            if move.get_piece_moved().get_location() == piece.get_location():
                return True
        return False

    def get_move_to_location(self, row, col):
        """Retrieve a move object for the destination cell if it is a legal move."""
        for move in self.legal_piece_moves:
            if move.get_destination() == (row, col):
                return move
        return None

    def highlight_eating_piece(self):
        player_pieces = self.state.board.white_pieces if self.color == WHITE else self.state.board.black_pieces
        for piece in player_pieces:
            if isinstance(piece, EatingPiece):
                self.selected_piece = piece
                self.legal_moves = self.state.get_piece_moves(piece)
                self.display.highlight_legal_moves(self.selected_piece, self.legal_moves)
                return
