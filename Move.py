from Pieces import Piece


class Move:
    def __init__(
        self,
        piece_moved: Piece,
        destination: tuple[int, int],
        pieces_eaten: list[Piece],
    ) -> None:
        self.piece_moved = piece_moved
        self.destination = destination
        self.pieces_eaten = pieces_eaten

    def get_piece_moved(self) -> Piece:
        return self.piece_moved

    def get_destination(self) -> tuple[int, int]:
        return self.destination

    def get_pieces_eaten(self) -> list[Piece]:
        return self.pieces_eaten

    def __repr__(self):
        return f"\n from: {self.piece_moved}, to {self.destination}, eaten: {self.pieces_eaten}"
