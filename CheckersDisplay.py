import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

from Constants import BOARD_SIZE, EMPTY, BLACK, WHITE


class CheckersDisplay:
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.root = tk.Tk()
        self.root.title("Checkers Game")

        # self.root.attributes('-fullscreen', True)

        self.player_selection_frame = None

        self.num_games = tk.IntVar(value=1)

        self.crown_image = Image.open("crown.png")
        self.crown_photos = []
        self.create_player_selection_screen()

    def create_player_selection_screen(self):
        if self.player_selection_frame:
            self.player_selection_frame.destroy()

        # Frame for player selection
        self.player_selection_frame = tk.Frame(self.root)
        self.player_selection_frame.pack(padx=20, pady=20)

        tk.Label(self.player_selection_frame,
                 text="Select Player 1",
                 font=('Verdana', 18, 'bold')).grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.player1_type = tk.StringVar(value='human')
        tk.Radiobutton(self.player_selection_frame,
                       text='Human',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='human').grid(row=1, column=0, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Random',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='random').grid(row=2, column=0, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Minimax',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='minimax').grid(row=3, column=0, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Reinforced Learning',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='rl').grid(row=4, column=0, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='DNN',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='dnn').grid(row=5, column=0, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='First Choice',
                       font=('Verdana', 16),
                       variable=self.player1_type,
                       value='first choice').grid(row=6, column=0, padx=10, pady=5, sticky='w')

        tk.Label(self.player_selection_frame,
                 text="Select Player 2",
                 font=('Verdana', 18, 'bold')).grid(row=0, column=1, padx=10, pady=10)
        self.player2_type = tk.StringVar(value='human')
        tk.Radiobutton(self.player_selection_frame,
                       text='Human',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='human').grid(row=1, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Random',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='random').grid(row=2, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Minimax',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='minimax').grid(row=3, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='Reinforced Learning',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='rl').grid(row=4, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='DNN',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='dnn').grid(row=5, column=1, padx=10, pady=5, sticky='w')
        tk.Radiobutton(self.player_selection_frame,
                       text='First Choice',
                       font=('Verdana', 16),
                       variable=self.player2_type,
                       value='first choice').grid(row=6, column=1, padx=10, pady=5, sticky='w')

        # Add label and entry for number of games
        tk.Label(self.player_selection_frame,
                 text="Number of Games:",
                 font=("Verdana", 16, 'bold')).grid(row=7, column=0, padx=(0, 10), pady=10, sticky="e")

        # Entry field with the default value displayed as "1"
        num_games_entry = tk.Entry(self.player_selection_frame, textvariable=self.num_games, font=("Verdana", 16),
                                   width=5)
        num_games_entry.grid(row=7, column=1, padx=(10, 0), pady=10, sticky="w")

        # Center the entire grid in the frame
        self.player_selection_frame.grid_columnconfigure(0, weight=1)
        self.player_selection_frame.grid_columnconfigure(1, weight=1)

        num_games_entry.focus_set()

        tk.Button(self.player_selection_frame,
                  text="Start Game",
                  font=('Verdana', 16),
                  command=self.start_game).grid(row=8, column=0, columnspan=2, pady=20)

    def start_game(self):

        try:
            num_games = self.num_games.get()

            # Check if the number of games is a positive integer greater than 0
            if num_games <= 0:
                raise ValueError("Number of games must be an integer greater than 0.")
        except (tk.TclError, ValueError) as e:
            tk.messagebox.showerror("Invalid Input", str(e))
            return

        self.game_manager.set_num_of_games(num_games)
        player1 = self.player1_type.get()
        player2 = self.player2_type.get()

        self.game_manager.init_game(player1, player2)

        self.player_selection_frame.pack_forget()

        self.create_game_screen()
        import threading
        game_thread = threading.Thread(target=self.game_manager.run_game_loop)
        game_thread.start()
        # self.game_manager.run_game_loop()

    def create_game_screen(self):
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack(fill='both', expand=True)

        # Timer label
        self.timer_label = tk.Label(self.game_frame, text="Timer: 00:00", font=("Verdana", 16))
        self.timer_label.pack(pady=10)

        # Turn label
        self.turn_label = tk.Label(self.game_frame, text="Turn: Player 1 (Red)", font=("Verdana", 14))
        self.turn_label.pack(pady=5)

        # Frame to hold the board and scores
        self.board_frame = tk.Frame(self.game_frame)
        self.board_frame.pack()

        # Player 1 score label
        self.player1_score_label = tk.Label(self.board_frame, text="Player 1 (BLUE) Score: 0", font=("Verdana", 16))
        self.player1_score_label.grid(row=0, column=0, padx=20, pady=10)

        # Canvas for the board
        self.canvas = tk.Canvas(self.board_frame, width=8 * 70, height=8 * 70, bg='white')
        self.canvas.grid(row=0, column=1)

        # Player 2 score label
        self.player2_score_label = tk.Label(self.board_frame, text="Player 2 (RED) Score: 0", font=("Verdana", 16))
        self.player2_score_label.grid(row=0, column=2, padx=20, pady=10)

        self.cell_size = 70  # Size of each cell in pixels

        # Draw the grid
        self.draw_grid()

        # Start the timer
        self.start_timer()

        self.message_label = tk.Label(self.game_frame, text='', font=("Verdana", 20, 'bold'), fg='red')
        self.message_label.pack(side="bottom", pady=10)

    def start_timer(self):
        self.timer_seconds = 0
        self.update_timer()

    def update_timer(self):
        if self.timer_label.winfo_exists():  # Check if the label exists
            minutes = self.timer_seconds // 60
            seconds = self.timer_seconds % 60
            self.timer_label.config(text=f"Timer: {minutes:02}:{seconds:02}")
            self.timer_seconds += 1
            self.root.after(1000, self.update_timer)

    def draw_grid(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x0 = col * self.cell_size
                y0 = row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                # Alternate colors
                color = 'white' if (row + col) % 2 == 0 else 'black'
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='black')

    def render_board(self):
        black_pieces = self.game_manager.game.board.black_pieces
        white_pieces = self.game_manager.game.board.white_pieces

        # Clear any existing pieces
        self.canvas.delete('piece')

        for piece in black_pieces:
            try:
                self.render_piece(piece, 'blue')

            except Exception as e:
                print(f"Error rendering black piece: {e}")

        for piece in white_pieces:
            try:
                self.render_piece(piece, 'red')

            except Exception as e:
                print(f"Error rendering white piece: {e}")

        self.switch_turn()

    def render_piece(self, piece, color):
        row, col = piece.get_location()

        x_center = col * self.cell_size + self.cell_size / 2
        y_center = (BOARD_SIZE - 1 - row) * self.cell_size + self.cell_size / 2
        piece_radius = self.cell_size / 3  # Radius of the pieces
        self.canvas.create_oval(
            x_center - piece_radius, y_center - piece_radius,
            x_center + piece_radius, y_center + piece_radius,
            fill=color, outline='black', tags='piece'
        )

        if piece.is_queen():
            crown_size = int(self.cell_size / 2)
            resized_crown = self.crown_image.resize((crown_size, crown_size))
            crown_photo = ImageTk.PhotoImage(resized_crown)  # Convert to Tkinter format
            self.crown_photos.append(
                crown_photo)  # Store the crown image to prevent it from being garbage collected
            crown_x = x_center - crown_size / 2
            crown_y = y_center - crown_size / 2
            self.canvas.create_image(crown_x, crown_y, image=crown_photo, anchor='nw', tags='piece')

    def switch_turn(self):
        if self.game_manager.game.current_player.color == BLACK:
            self.turn_label.config(text="Turn: Player 1 (Blue)")
        else:
            self.turn_label.config(text="Turn: Player 2 (Red)")

    def highlight_legal_moves(self, piece, legal_moves):
        """Highlights the legal moves for the selected piece."""
        # First, clear any previous highlights
        self.clear_highlights()

        self.highlight_selected_piece(piece)

        # Iterate over all the legal moves and highlight the cells
        for move in legal_moves:
            destination = move.get_destination()
            row, col = destination
            row = BOARD_SIZE - 1 - row

            # Highlight the legal move destination (for example, with a light green color)
            self.canvas.create_rectangle(
                col * self.cell_size + 4,
                row * self.cell_size + 4,
                (col + 1) * self.cell_size - 4,
                (row + 1) * self.cell_size - 4,
                outline="#00FF00",  # Dark green outline
                width=6,  # Thicker outline for better visibility
                tags="highlight"  # Tag for clearing later
            )

    def highlight_selected_piece(self, piece):
        """Highlights the selected piece by drawing a border around it."""
        # Get the location of the selected piece
        row, col = piece.get_location()
        row = BOARD_SIZE - 1 - row

        # Calculate the center and radius of the piece
        piece_x = col * self.cell_size + self.cell_size // 2
        piece_y = row * self.cell_size + self.cell_size // 2
        piece_radius = self.cell_size // 3

        # Highlight the border of the selected piece by drawing a circle around it
        self.canvas.create_oval(
            piece_x - piece_radius,  # Border padding
            piece_y - piece_radius,
            piece_x + piece_radius,
            piece_y + piece_radius,
            outline="lightblue",  # Blue outline for selection
            width=5,  # Thickness of the border
            tags="highlight"  # Tag for clearing later
        )

    def clear_highlights(self):
        """Clears all highlighted cells (legal move indications)."""
        # Simply delete any objects with the "highlight" tag
        self.canvas.delete("highlight")

    def show_end_result(self, player1_score, player2_score, num_ties):
        # Create a new window for the end result
        end_result_window = tk.Toplevel(self.game_frame)
        end_result_window.title("Game Over")

        # Get the dimensions of the current window
        window_width = self.game_frame.winfo_width()
        window_height = self.game_frame.winfo_height()

        # Set the dimensions and position of the end result window relative to the main window
        end_result_window.geometry(
            f"{window_width // 2}x{window_height // 2}"
            f"+{self.game_frame.winfo_x() + window_width // 4}"
            f"+{self.game_frame.winfo_y() + window_height // 4}")

        # Create labels for the scores
        tk.Label(end_result_window, text="Game Over!", font=("Arial", 16)).pack(pady=10)

        # Create a frame to hold the scores
        score_frame = tk.Frame(end_result_window)
        score_frame.pack(pady=20)

        # Player 1 Score
        player1_label = tk.Label(score_frame, text=f"Red Player Score: {player1_score}", font=("Arial", 14))
        player1_label.grid(row=0, column=0, padx=(10, 40))

        # Ties
        ties_label = tk.Label(score_frame, text=f"Ties: {num_ties}", font=("Arial", 14))
        ties_label.grid(row=0, column=1)

        # Player 2 Score
        player2_label = tk.Label(score_frame, text=f"Blue Player Score: {player2_score}", font=("Arial", 14))
        player2_label.grid(row=0, column=2, padx=(40, 10))

        button_frame = tk.Frame(end_result_window)
        button_frame.pack(pady=10)

        # Button to play again
        play_again_button = tk.Button(button_frame, text="Play Again", font=("Arial", 12),
                                      command=lambda: self.play_again(end_result_window))
        play_again_button.grid(row=0, column=0, padx=20)

        # Button to return to player selection
        player_selection_button = tk.Button(button_frame, text="Return to Player Selection", font=("Arial", 12),
                                            command=lambda: self.return_to_player_selection(end_result_window))
        player_selection_button.grid(row=0, column=1, padx=20)

    def play_again(self, end_result_window):
        # Destroy the end result window
        end_result_window.destroy()
        self.game_frame.destroy()

        self.num_games.set(1)

        self.start_game()

    def return_to_player_selection(self, end_result_window):
        # Destroy the end result window
        end_result_window.destroy()
        self.game_frame.destroy()

        self.game_manager.reset_scores()

        # Return to the player selection screen
        # self.player_selection_frame.pack(fill=tk.BOTH, expand=True)
        self.create_player_selection_screen()

    def update_scores(self, player1_score, player2_score):
        self.player1_score_label.config(text=f"Player 1 (BLUE) score: {player1_score}")
        self.player2_score_label.config(text=f"Player 2 (RED) score: {player2_score}")

    def display_message_beneath_board(self, message):
        self.message_label.config(text=message)

    def hide_message(self):
        self.message_label.config(text='')
