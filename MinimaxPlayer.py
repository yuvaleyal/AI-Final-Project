from typing import List

from Constants import WHITE, BOARD_SIZE, NOT_OVER_YET
from Pieces import Piece
from Player import Player
from State import State


class MinimaxPlayer(Player):
    def __init__(self, color: int, depth=6) -> None:
        super().__init__(color)
        self.depth = depth

    def set_depth(self, new_depth):
        self.depth = new_depth

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

        if current_depth == self.depth or state.is_over() != NOT_OVER_YET:
            return self.evaluate(state), None

        value = float('-inf')
        best_move = None

        for move in legal_moves:
            successor_state = state.generate_successor(move)
            successor_value, _ = self._min_value(successor_state, current_depth + 1, alpha, beta)

            if successor_value > value:
                value = successor_value
                best_move = move

            alpha = max(alpha, value)

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

        if current_depth == self.depth or state.is_over() != NOT_OVER_YET:
            return self.evaluate(state), None

        value = float('inf')
        best_move = None

        for move in legal_moves:
            successor_state = state.generate_successor(move)
            successor_value, _ = self._max_value(successor_state, current_depth + 1, alpha, beta)

            if successor_value < value:
                value = successor_value
                best_move = move

            beta = min(beta, value)

            if value <= alpha:
                break

        return value, best_move

    def evaluate(self, state: State) -> float:
        """
        Evaluates the current state of the board for the checkers game.

        This evaluation function takes into account material, positioning, captures,
        and threats, aiming to improve the AI's decision-making.
        :param state: The current state of the game.
        :return: A score representing the evaluation of the board.
        """
        piece_weight = 10
        king_weight = 15
        positional_weight = 1
        capture_weight = 20
        threat_weight = -12

        board = state.board
        player_pieces = board.get_pieces(self.color)
        opponent_pieces = board.get_pieces(-self.color)

        player_queens = self._num_of_queens(player_pieces)
        opponent_queens = self._num_of_queens(opponent_pieces)
        material_score = (
            piece_weight * ((len(player_pieces) - player_queens) - (len(opponent_pieces) - opponent_queens)) +
            king_weight * (player_queens - opponent_queens)
        )

        positional_score = 0
        for piece in player_pieces:
            row, col = piece.get_location()
            positional_score += self._get_positional_value(row, col)

        for piece in opponent_pieces:
            row, col = piece.get_location()
            positional_score -= self._get_positional_value(row, col)

        capture_score = 0
        player_moves = state.find_all_moves()
        opponent_moves = State(state.board, -self.color).find_all_moves()

        for move in player_moves:
            if len(move.get_pieces_eaten()) > 0:
                capture_score += capture_weight

        for move in opponent_moves:
            if len(move.get_pieces_eaten()) > 0:
                capture_score -= capture_weight

        threatened_player_pieces = set()
        for move in opponent_moves:
            for piece in move.get_pieces_eaten():
                threatened_player_pieces.add(piece)

        threatened_opponent_pieces = set()
        for move in player_moves:
            for piece in move.get_pieces_eaten():
                threatened_opponent_pieces.add(piece)

        threat_score = threat_weight * len(threatened_player_pieces) - threat_weight * len(threatened_opponent_pieces)
        evaluation_score = material_score + positional_weight * positional_score + capture_score + threat_score

        return evaluation_score

    def _get_positional_value(self, row, col) -> int:
        """
        Assigns a positional value to a piece based on its location on the board.
        :param row: The row of the piece.
        :param col: The column of the piece.
        :return: A score representing the positional value.
        """
        if self.color == WHITE:
            return BOARD_SIZE - 1 - row
        else:
            return row

    def _num_of_queens(self, pieces: List[Piece]) -> int:
        result = 0
        for piece in pieces:
            if piece.is_queen():
                result += 1
        return result




