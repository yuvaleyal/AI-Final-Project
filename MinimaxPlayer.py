from copy import deepcopy
from typing import List

from Board import Board
from Constants import WHITE, BOARD_SIZE, BLACK
from Move import Move
from Pieces import Piece, QueenPiece, RegularPiece
from Player import Player
from State import State


class MinimaxPlayer(Player):
    def __init__(self, color: int) -> None:
        super().__init__(color)

    def make_move(self, state: State) -> State:
        """Makes a move using the Minimax algorithm with alpha-beta pruning."""
        value, move = self._max_value(state, 0, float('-inf'), float('inf'))
        return state.next_state(move)

    def _max_value(self, state: State, current_depth: int, alpha: float, beta: float):
        """
        The maximizing player function in the Minimax algorithm.
        :param state: The current state of the game.
        :param current_depth: The current depth of the search tree.
        :param alpha: The best value that the maximizing player can guarantee so far.
        :param beta: The best value that the minimizing player can guarantee so far.
        :return: The maximum value and corresponding move.
        """
        legal_moves = state.find_all_moves()

        # Terminal condition or maximum depth reached
        if current_depth == 3 or not legal_moves:
            return self.evaluate(state), None

        value = float('-inf')
        best_move = None

        for move in legal_moves:
            successor_state = self.generate_successor(state, move)
            successor_value, _ = self._min_value(successor_state, current_depth, alpha, beta)

            if successor_value > value:
                value = successor_value
                best_move = move

            alpha = max(alpha, value)

            # Prune the remaining branches
            if value >= beta:
                break

        return value, best_move

    def _min_value(self, state: State, current_depth: int, alpha: float, beta: float):
        """
        The minimizing player function in the Minimax algorithm.
        :param state: The current state of the game.
        :param current_depth: The current depth of the search tree.
        :param alpha: The best value that the maximizing player can guarantee so far.
        :param beta: The best value that the minimizing player can guarantee so far.
        :return: The minimum value and corresponding move.
        """
        legal_moves = state.find_all_moves()

        # Terminal condition or maximum depth reached
        if current_depth == 3 or not legal_moves:
            return self.evaluate(state), None

        value = float('inf')
        best_move = None

        for move in legal_moves:
            successor_state = self.generate_successor(state, move)
            successor_value, _ = self._max_value(successor_state, current_depth + 1, alpha, beta)

            if successor_value < value:
                value = successor_value
                best_move = move

            beta = min(beta, value)

            # Prune the remaining branches
            if value <= alpha:
                break

        return value, best_move

    def evaluate(self, state: State) -> float:
        """
        Evaluates the current state of the board for the checkers game.

        This evaluation function is based on the material, position, and mobility of pieces.
        :param state: The current state of the game.
        :return: A score representing the evaluation of the board.
        """
        # Weights for different factors
        piece_weight = 8  # Weight for a regular piece
        king_weight = 10  # Weight for a king piece
        positional_weight = 1  # Positional value (for central and advanced pieces)
        mobility_weight = 0.1  # Weight for the number of legal moves

        # Get the board from the state
        board = state.board
        player_pieces = board.get_pieces(self.color)
        opponent_pieces = board.get_pieces(state.last_player)

        # Material count: count pieces and kings
        player_queens = self.num_of_queens(player_pieces)
        opponent_queens = self.num_of_queens(opponent_pieces)
        material_score = (
            piece_weight * ((len(player_pieces) - player_queens) - (len(opponent_pieces) - opponent_queens)) +
            king_weight * (player_queens - opponent_queens)
        )

        # Positional score: favor central positions or advanced pieces
        positional_score = 0
        # for piece in player_pieces:
        #     row, col = piece.get_location()
        #     positional_score += self.get_positional_value(row, col)
        #
        # for piece in opponent_pieces:
        #     row, col = piece.get_location()
        #     positional_score -= self.get_positional_value(row, col)

        # Mobility score: favor the player with more legal moves
        player_moves = len(state.find_all_moves())
        mobility_score = mobility_weight * (player_moves)

        # Total evaluation score
        evaluation_score = material_score + positional_weight * positional_score + mobility_score

        return evaluation_score

    def get_positional_value(self, row, col) -> int:
        """
        Assigns a positional value to a piece based on its location on the board.
        :param row: The row of the piece.
        :param col: The column of the piece.
        :return: A score representing the positional value.
        """
        # Favor central positions and advanced rows
        # These values can be fine-tuned for better performance.
        # Pieces closer to promotion (top rows for white, bottom rows for black) get higher scores
        if self.color == WHITE:  # White pieces
            return BOARD_SIZE - 1 - row  # More advanced rows (higher row index) get higher scores
        else:  # Black pieces
            return row  # More advanced rows (lower row index) get higher scores

    def num_of_queens(self, pieces: List[Piece]) -> int:
        result = 0
        for piece in pieces:
            if piece.is_queen():
                result += 1
        return result

    # This mehtod should be in State class:
    def generate_successor(self, state: State, move) -> State:
        # Create a deep copy of the current state
        state_copy = deepcopy(state)

        piece_to_move = None
        for piece in state_copy.board.get_pieces(-state_copy.last_player):
            if piece.get_location() == move.get_piece_moved().get_location():
                piece_to_move = piece
                break

        eaten_pieces = []
        eaten_pieces_locations = [piece.get_location() for piece in move.get_pieces_eaten()]

        for piece in state_copy.board.get_pieces(state_copy.last_player):
            if piece.get_location() in eaten_pieces_locations:
                eaten_pieces.append(piece)

        move_copy = Move(piece_to_move, move.get_destination(), eaten_pieces)

        state_copy.board.make_move(move_copy)

        # Return the successor state
        return State(state_copy.board, -state_copy.last_player)




