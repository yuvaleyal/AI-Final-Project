# import dash
# from dash import dcc, html, Input, Output, State
# import dash_bootstrap_components as dbc
# from dash.exceptions import PreventUpdate
#
# from Constants import BOARD_SIZE, BLACK
#
#
# class CheckersDisplay:
#     def __init__(self, game_manager):
#         self.game_manager = game_manager
#         self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#         self.init_layout()
#         self.init_callbacks()
#
#     def init_layout(self):
#         self.app.layout = html.Div([
#             html.H1("Checkers Game", style={"textAlign": "center"}),
#
#             # Home Screen
#             html.Div(id='home-screen', children=[
#                 html.H2("Select Players", style={"textAlign": "center"}),
#
#                 html.Div([
#                     html.Div([
#                         html.Label("Player 1", style={'fontSize': '22px', 'textAlign': 'center', 'display': 'block'}),
#                         dcc.RadioItems(
#                             id='player1-radiolist',
#                             options=[
#                                 {'label': ' Random', 'value': 'random'},
#                                 {'label': ' Minimax', 'value': 'minimax'}
#                             ],
#                             value='random',  # Default value
#                             labelStyle={'display': 'block', 'textAlign': 'left', 'paddingLeft': '150px'}
#                         )
#                     ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
#
#                     html.Div([
#                         html.Label("Player 2", style={'fontSize': '22px', 'textAlign': 'center', 'display': 'block'}),
#                         dcc.RadioItems(
#                             id='player2-radiolist',
#                             options=[
#                                 {'label': ' Random', 'value': 'random'},
#                                 {'label': ' Minimax', 'value': 'minimax'}
#                             ],
#                             value='random',  # Default value
#                             labelStyle={'display': 'block', 'textAlign': 'left', 'paddingLeft': '150px'}
#                         )
#                     ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'})
#                 ], style={'textAlign': 'center', 'width': '100%'}),
#
#                 html.Div(
#                     dbc.Button("Start Game", id="start-game-btn", color="primary", style={"marginTop": "20px"}),
#                     style={"textAlign": "center"}
#                 ),
#             ]),
#
#             html.Div(id='game-screen', style={'display': 'none'}, children=[
#                 html.Div(id="checkers-board",
#                          style={
#                              'display': 'flex',
#                              'justifyContent': 'center',
#                              'alignItems': 'center',
#                              'height': '650px',
#                              'width': '650px',
#                              'margin': '0 auto'
#                          }
#                          ),
#                 dcc.Store(id='game-state', data=None),
#                 html.Div(id="message", style={"textAlign": "center", "marginTop": "20px"}),
#                 dcc.Store(id='game-trigger', data=''),
#                 dcc.Interval(id='update-interval', interval=1000, n_intervals=0)
#             ])
#         ])
#
#     def init_callbacks(self):
#         @self.app.callback(
#             [Output('home-screen', 'style'),
#              Output('game-screen', 'style'),
#              # Output('checkers-board', 'children'),
#              Output('game-trigger', 'data')],
#             [Input('start-game-btn', 'n_clicks')],
#             [State('player1-radiolist', 'value'),
#              State('player2-radiolist', 'value')]
#         )
#         def start_game(n_clicks, player1, player2):
#             if n_clicks is None:
#                 raise PreventUpdate
#
#             # Initialize the game
#             self.game_manager.init_game(player1, player2)
#             # board_table = self.update_board()
#
#             # Set the game-trigger to 'start' after initializing the game
#             return {'display': 'none'}, {'display': 'block'}, 'start'
#
#         @self.app.callback(
#             Input('game-trigger', 'data'),
#             prevent_initial_call=True
#         )
#         def run_game_loop(trigger):
#             if trigger == 'start':
#                 print("Game loop should start now")
#                 self.game_manager.run_game_loop()
#                 return dash.no_update
#
#         @self.app.callback(
#             Output('game-state', 'data'),
#             Input('update-interval', 'n_intervals')
#         )
#         def update_game_state(n_intervals):
#             # Update the game state every time the interval triggers
#             return self.get_board_state()
#
#         @self.app.callback(
#             Output('checkers-board', 'children'),
#             Input('game-state', 'data')
#         )
#         def update_board_from_state(board_state):
#             if board_state is None:
#                 raise PreventUpdate
#
#             # Get the positions of black and white pieces from the board state
#             black_pieces = board_state.get('black_pieces', [])
#             white_pieces = board_state.get('white_pieces', [])
#
#             # Create an HTML table for the checkers board
#             board_html = []
#
#             for row in range(BOARD_SIZE):
#                 row_cells = []
#                 for col in range(BOARD_SIZE):
#                     # Determine the color of the cell
#                     color = 'white' if (row + col) % 2 == 0 else 'black'
#                     cell_style = {
#                         'backgroundColor': color,
#                         'width': '80px',
#                         'height': '80px',
#                         'textAlign': 'center',
#                         'verticalAlign': 'middle',
#                         'border': '1px solid black'
#                     }
#
#                     # Initialize the cell content
#                     cell_content = None
#
#                     # Check if a black or white piece is at this position
#                     if (row, col) in black_pieces:
#                         # Black piece
#                         piece_color = 'blue'
#                         cell_content = html.Div(style={
#                             'width': '75%',
#                             'height': '75%',
#                             'borderRadius': '50%',
#                             'backgroundColor': piece_color,
#                             'margin': 'auto'
#                         })
#                     elif (row, col) in white_pieces:
#                         # White piece
#                         piece_color = 'red'
#                         cell_content = html.Div(style={
#                             'width': '75%',
#                             'height': '75%',
#                             'borderRadius': '50%',
#                             'backgroundColor': piece_color,
#                             'margin': 'auto'
#                         })
#
#                     # Append the cell to the row
#                     row_cells.append(html.Td(cell_content, style=cell_style))
#
#                 # Append the row to the board
#                 board_html.append(html.Tr(row_cells))
#
#             # Return the HTML table to render the board
#             return html.Table(board_html, style={'margin': 'auto', 'borderCollapse': 'collapse'})
#
#     def get_board_state(self):
#         if self.game_manager is None:
#             return {}
#         if self.game_manager.game is None:
#             return {}
#
#         black_pieces = [(piece.get_location(), piece.get_player()) for piece in self.game_manager.game.board.black_pieces]
#         white_pieces = [(piece.get_location(), piece.get_player()) for piece in self.game_manager.game.board.white_pieces]
#
#         return {
#             'black_pieces': black_pieces,
#             'white_pieces': white_pieces
#         }
#
#     # def render_board(self):
#     #     # Method to update game-state after each move
#     #     self.app.callback_map['checkers-board.children']['state'] = self.get_board_state()
#     #     print(self.app.callback_map['checkers-board.children']['state'])
#
#         # @self.app.callback(
#         #     Output('checkers-board', 'children'),
#         #     Input('update-interval', 'n_intervals')
#         # )
#         # def update_board(n_intervals):
#         #     if self.game_manager.game is None:
#         #         return dash.no_update
#         #     if self.game_manager.game.needs_update:
#         #         self.game_manager.game.needs_update = False
#         #         return self.update_board()
#         #     return dash.no_update
#
# # app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
# #
# # # Create a checkers game object
# # game = CheckersGame()
# #
# # # Player type selection screen
# # selection_layout = html.Div(id='selection-screen', children=[
# #     html.H1("Choose Your Opponent"),
# #     dcc.RadioItems(
# #         id='player-selection',
# #         options=[
# #             {'label': 'Human', 'value': 'human'},
# #             {'label': 'Random AI', 'value': 'random_ai'},
# #             {'label': 'Minimax with Alpha-Beta Pruning', 'value': 'minimax'},
# #             {'label': 'Reinforcement Learning', 'value': 'rl'},
# #             {'label': 'Neural Networks', 'value': 'nn'}
# #         ],
# #         value='random_ai',  # Default random player
# #         labelStyle={'display': 'block'}
# #     ),
# #     html.Button('Start Game', id='start-button', n_clicks=0)
# # ])
# #
# #
# # # The game screen
# # def game_layout(selected_player):
# #     board = game.get_board_state()
# #     return html.Div([
# #         html.H2(f"Opponent: {selected_player}"),
# #         html.Div(id='current-player', children=f"Current Player: {game.get_current_player()}"),
# #         html.Div(id='captured-pieces', children="Captured - White: 0, Black: 0"),
# #         html.Div([
# #             html.Table([
# #                 html.Tr([
# #                     html.Td(
# #                         html.Button(board[i][j], id=f'cell-{i}-{j}', className='board-cell',
# #                                     style={
# #                                         'width': '50px',
# #                                         'height': '50px',
# #                                         'background-color': 'black' if (i + j) % 2 == 1 else 'white',
# #                                         'color': 'darkgray' if board[i][j] == 'B' else 'white'
# #                                     }),
# #                         style={'border': '1px solid black'}
# #                     )
# #                     for j in range(8)
# #                 ]) for i in range(8)
# #             ])
# #         ]),
# #         html.Div([
# #             html.Div(id='game-timer', children="Game Timer: 0:00", style={'margin-top': '20px'}),
# #             html.Div(id='player-timer', children="Player Timer: 0:00", style={'margin-top': '10px'}),
# #         ])
# #     ])
# #
# #
# # # Defining the initial content in main-container
# # app.layout = html.Div(id='main-container', children=[selection_layout])
# #
# #
# # @app.callback(
# #     Output('main-container', 'children'),
# #     Input('start-button', 'n_clicks'),
# #     State('player-selection', 'value')
# # )
# # def start_game(n_clicks, selected_player):
# #     if n_clicks > 0:
# #         game.__init__()  # Restart the game
# #         return [game_layout(selected_player)]
# #     return dash.no_update
# #
# #
# # @app.callback(
# #     [Output({'type': 'board-cell', 'index': ALL}, 'children'),
# #      Output('current-player', 'children')],
# #     [Input({'type': 'board-cell', 'index': ALL}, 'n_clicks')],
# #     [State('main-container', 'children')]
# # )
# # def update_board(cell_clicks, main_container):
# #     ctx = dash.callback_context
# #     if not ctx.triggered or not any(cell_clicks):
# #         return [dash.no_update] * 64, dash.no_update  # Return a list with 64 values for all slots
# #
# #     # Finding the cell that was clicked
# #     cell_id = ctx.triggered[0]['prop_id'].split('.')[0]
# #     row, col = map(int, cell_id.split('-')[1:])
# #
# #     # Making a move
# #     opponent_type = main_container[0]['props']['children'][1]['props']['children'][1]['props']['value']
# #     game.play_turn(row, col, opponent_type)
# #
# #     # Updating the state of the buttons on the panel
# #     updated_cells = []
# #     board = game.get_board_state()
# #     for i in range(8):
# #         for j in range(8):
# #             updated_cells.append(board[i][j])
# #
# #     return updated_cells, f"Current Player: {game.get_current_player()}"


import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

from Constants import BOARD_SIZE, EMPTY, BLACK, WHITE
from Move import Move


class CheckersDisplay:
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.root = tk.Tk()
        self.root.title("Checkers Game")

        self.player_selection_frame = None

        self.num_games = tk.IntVar(value=1)
        self.display_board = tk.BooleanVar(value=True)

        self.crown_image = Image.open("crown.png")
        self.crown_photos = []
        self.create_player_selection_screen()

    def create_player_selection_screen(self):
        if self.player_selection_frame:
            self.player_selection_frame.destroy()
        # Frame for player selection
        self.player_selection_frame = tk.Frame(self.root)
        self.player_selection_frame.pack(padx=20, pady=20)

        tk.Label(self.player_selection_frame, text="Select Player 1").grid(row=0, column=0, padx=10, pady=10)
        self.player1_type = tk.StringVar(value='human')
        tk.Label(self.player_selection_frame, text="Select Player 1").grid(row=0, column=0, padx=10, pady=10)
        tk.Radiobutton(self.player_selection_frame, text='Human', variable=self.player1_type, value='human').grid(
            row=1, column=0, padx=10, pady=5)
        tk.Radiobutton(self.player_selection_frame, text='Random', variable=self.player1_type, value='random').grid(
            row=2, column=0, padx=10, pady=5)
        tk.Radiobutton(self.player_selection_frame, text='Minimax', variable=self.player1_type, value='minimax').grid(
            row=3, column=0, padx=10, pady=5)

        tk.Label(self.player_selection_frame, text="Select Player 2").grid(row=0, column=1, padx=10, pady=10)
        self.player2_type = tk.StringVar(value='human')
        tk.Radiobutton(self.player_selection_frame, text='Human', variable=self.player2_type, value='human').grid(
            row=1, column=1, padx=10, pady=5)
        tk.Radiobutton(self.player_selection_frame, text='Random', variable=self.player2_type, value='random').grid(
            row=2, column=1, padx=10, pady=5)
        tk.Radiobutton(self.player_selection_frame, text='Minimax', variable=self.player2_type, value='minimax').grid(
            row=3, column=1, padx=10, pady=5)

        # Add label and entry for number of games
        tk.Label(self.player_selection_frame, text="Number of Games:", font=("Arial", 14)).grid(row=4,
                                                                                                column=0,
                                                                                                pady=20)

        # Entry field with the default value displayed as "1"
        num_games_entry = tk.Entry(self.player_selection_frame, textvariable=self.num_games, font=("Arial", 14),
                                   width=5)
        num_games_entry.grid(row=4, column=1, pady=20)

        num_games_entry.focus_set()

        toggle_button = tk.Checkbutton(self.player_selection_frame, text="Display Board", variable=self.display_board)
        toggle_button.grid(row=5, pady=20)

        tk.Button(self.player_selection_frame, text="Start Game", command=self.start_game).grid(row=6, column=0,
                                                                                                columnspan=2, pady=20)

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

        try:
            display_board = self.display_board.get()

            if not display_board and (player1 == "human" or player2 == "human"):
                raise ValueError("Board must be displayed when human is playing.")
        except (tk.TclError, ValueError) as e:
            tk.messagebox.showerror("Invalid Input", str(e))
            return

        self.game_manager.set_board_display(display_board)

        self.game_manager.init_game(player1, player2)

        self.player_selection_frame.pack_forget()
        if display_board:
            self.create_game_screen()
        else:
            self.create_progress_bar(num_games)
        import threading
        game_thread = threading.Thread(target=self.game_manager.run_game_loop)
        game_thread.start()
        # self.game_manager.run_game_loop()

    def create_progress_bar(self, num_games):
        """Create and display the progress bar for non-board games."""
        self.progress_frame = tk.Frame(self.root)
        self.progress_frame.pack(padx=20, pady=20)

        # Progress label
        self.progress_label = tk.Label(self.progress_frame, text="Playing games...", font=("Helvetica", 14))
        self.progress_label.pack(pady=10)

        # Create progress bar
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal', length=300, mode='determinate')
        self.progress_bar.pack(pady=20)

        # Set the maximum value of the progress bar to the number of games
        self.progress_bar['maximum'] = num_games

        # Initialize the progress bar value to 0
        self.progress_bar['value'] = 0

    def update_progress_bar(self, completed_games):
        """Update the progress bar based on the number of completed games."""
        if hasattr(self, 'progress_bar'):
            self.progress_bar['value'] = completed_games

            # Update the label to show the progress
            self.progress_label.config(text=f"Completed {completed_games} out of {self.num_games.get()} games")

            # Check if all games are completed
            if completed_games == self.num_games.get():
                self.progress_label.config(text="All games completed!")

    def create_game_screen(self):
        self.game_frame = tk.Frame(self.root)
        self.game_frame.pack(fill='both', expand=True)

        # Timer label
        self.timer_label = tk.Label(self.game_frame, text="Timer: 00:00", font=("Helvetica", 16))
        self.timer_label.pack(pady=10)

        # Turn label
        self.turn_label = tk.Label(self.game_frame, text="Turn: Player 1 (Red)", font=("Helvetica", 14))
        self.turn_label.pack(pady=5)

        # Frame to hold the board and scores
        self.board_frame = tk.Frame(self.game_frame)
        self.board_frame.pack()

        # Player 1 score label
        self.player1_score_label = tk.Label(self.board_frame, text="Player 1 (Black) Score: 0", font=("Helvetica", 12))
        self.player1_score_label.grid(row=0, column=0, padx=20, pady=10)

        # Canvas for the board
        self.canvas = tk.Canvas(self.board_frame, width=8 * 60, height=8 * 60, bg='white')
        self.canvas.grid(row=0, column=1)

        # Player 2 score label
        self.player2_score_label = tk.Label(self.board_frame, text="Player 2 (White) Score: 0", font=("Helvetica", 12))
        self.player2_score_label.grid(row=0, column=2, padx=20, pady=10)

        self.board = [[None] * 8 for _ in range(8)]
        self.pieces = [[None] * 8 for _ in range(8)]

        self.cell_size = 60  # Size of each cell in pixels

        # Draw the grid
        self.draw_grid()

        # Draw the pieces
        self.render_board()

        # Start the timer
        self.start_timer()

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
                color = 'black' if (row + col) % 2 == 0 else 'white'
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='black')

    def render_board(self):
        black_pieces = self.game_manager.game.board.black_pieces
        white_pieces = self.game_manager.game.board.white_pieces

        # Clear any existing pieces
        self.canvas.delete('piece')

        counter = 0

        for piece in black_pieces:
            try:
                row, col = piece.get_location()

                x_center = col * self.cell_size + self.cell_size / 2
                y_center = row * self.cell_size + self.cell_size / 2
                piece_radius = self.cell_size / 3  # Radius of the pieces
                self.canvas.create_oval(
                    x_center - piece_radius, y_center - piece_radius,
                    x_center + piece_radius, y_center + piece_radius,
                    fill='blue', outline='black', tags='b' + str(counter)
                )
                self.pieces[row][col] = 'b' + str(counter)
                counter += 1
            except Exception as e:
                print(f"Error rendering black piece: {e}")

        counter = 0
        for piece in white_pieces:
            try:
                row, col = piece.get_location()

                x_center = col * self.cell_size + self.cell_size / 2
                y_center = row * self.cell_size + self.cell_size / 2
                piece_radius = self.cell_size / 3  # Radius of the pieces
                self.canvas.create_oval(
                    x_center - piece_radius, y_center - piece_radius,
                    x_center + piece_radius, y_center + piece_radius,
                    fill='red', outline='black', tags='w' + str(counter)
                )
                self.pieces[row][col] = 'w' + str(counter)
                counter += 1
            except Exception as e:
                print(f"Error rendering white piece: {e}")

    def update_board(self, start_move_location, destination, pieces_eaten, color):
        # Remove the pieces eaten by the move
        # if len(pieces_eaten):
        for piece in pieces_eaten:
            row, col = piece.get_location()
            piece_id = self.pieces[row][col]
            if piece_id:  # Check if a piece exists in the location
                self.canvas.delete(piece_id)
                self.pieces[row][col] = None
            # row, col = piece_eaten.get_location()
            # piece_id = self.pieces[row][col]
            # if piece_id:  # Check if a piece exists in the location
            #     self.canvas.delete(piece_id)
            #     self.pieces[row][col] = None

        # Remove the moved piece from its original location

        row, col = start_move_location
        piece_id = self.pieces[row][col]
        if piece_id:  # Check if a piece exists in the location
            self.canvas.delete(piece_id)
            self.pieces[row][col] = None

        # Draw the cell background again at the new location
        new_row, new_col = destination
        x0 = new_col * self.cell_size
        y0 = new_row * self.cell_size
        x1 = x0 + self.cell_size
        y1 = y0 + self.cell_size

        # Redraw the cell background color based on its position
        # color = 'white' if (new_row + new_col) % 2 == 0 else 'black'
        self.canvas.create_rectangle(x0, y0, x1, y1, fill='black', outline='black')

        # Add the moved piece to its new location
        x_center_new = new_col * self.cell_size + self.cell_size / 2
        y_center_new = new_row * self.cell_size + self.cell_size / 2
        piece_radius = self.cell_size / 3  # Radius of the pieces

        # Redraw the moved piece at its new location
        if color == BLACK:
            self.canvas.create_oval(
                x_center_new - piece_radius, y_center_new - piece_radius,
                x_center_new + piece_radius, y_center_new + piece_radius,
                fill='blue', outline='black', tags=piece_id
            )
        else:
            self.canvas.create_oval(
                x_center_new - piece_radius, y_center_new - piece_radius,
                x_center_new + piece_radius, y_center_new + piece_radius,
                fill='red', outline='black', tags=piece_id
            )

        # If the piece is a queen, draw the crown
        if self.game_manager.game.board.get_piece((new_row, new_col)).is_queen():
            crown_size = int(self.cell_size / 2)
            resized_crown = self.crown_image.resize((crown_size, crown_size))
            crown_photo = ImageTk.PhotoImage(resized_crown)  # Convert to Tkinter format
            self.crown_photos.append(crown_photo)  # Store the crown image to prevent it from being garbage collected
            crown_x = x_center_new - crown_size / 2
            crown_y = y_center_new - crown_size / 2
            self.canvas.create_image(crown_x, crown_y, image=crown_photo, anchor='nw', tags=piece_id)

            # Store the new piece ID in the pieces matrix
        self.pieces[new_row][new_col] = piece_id

        self.switch_turn()

    def switch_turn(self):
        if self.game_manager.game.current_player.color == BLACK:
            self.turn_label.config(text="Turn: Player 2 (Blue)")
        else:
            self.turn_label.config(text="Turn: Player 1 (Red)")

    def on_button_click(self, row, col):
        # Handle button click events here
        print(f'Button clicked at {row}, {col}')

    def highlight_legal_moves(self, piece, legal_moves):
        """Highlights the legal moves for the selected piece."""
        # First, clear any previous highlights
        self.clear_highlights()

        self.highlight_selected_piece(piece)

        # Iterate over all the legal moves and highlight the cells
        for move in legal_moves:
            destination = move.get_destination()
            row, col = destination

            # Highlight the legal move destination (for example, with a light green color)
            self.canvas.create_rectangle(
                col * self.cell_size,
                row * self.cell_size,
                (col + 1) * self.cell_size,
                (row + 1) * self.cell_size,
                outline="#00FF00",  # Dark green outline
                width=6,  # Thicker outline for better visibility
                tags="highlight"  # Tag for clearing later
            )

    def highlight_selected_piece(self, piece):
        """Highlights the selected piece by drawing a border around it."""
        # Get the location of the selected piece
        row, col = piece.get_location()

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
            width=6,  # Thickness of the border
            tags="highlight"  # Tag for clearing later
        )

    def clear_highlights(self):
        """Clears all highlighted cells (legal move indications)."""
        # Simply delete any objects with the "highlight" tag
        self.canvas.delete("highlight")

    def show_end_result(self, player1_score, player2_score, num_ties):
        # Create a new window for the end result
        if self.display_board.get():
            self.end_screen_root = self.game_frame
        else:
            self.end_screen_root = self.progress_frame
        end_result_window = tk.Toplevel(self.end_screen_root)
        end_result_window.title("Game Over")

        # Get the dimensions of the current window
        # window_width = self.end_screen_root.winfo_width()
        # window_height = self.end_screen_root.winfo_height()
        #
        # # Set the dimensions and position of the end result window relative to the main window
        # end_result_window.geometry(
        #     f"{window_width // 2}x{window_height // 2}"
        #     f"+{self.end_screen_root.winfo_x() + window_width // 4}"
        #     f"+{self.end_screen_root.winfo_y() + window_height // 4}")

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
        self.end_screen_root.destroy()

        self.num_games.set(1)

        self.start_game()

    def return_to_player_selection(self, end_result_window):
        # Destroy the end result window
        end_result_window.destroy()
        self.end_screen_root.destroy()

        self.game_manager.reset_scores()

        # Return to the player selection screen
        # self.player_selection_frame.pack(fill=tk.BOTH, expand=True)
        self.create_player_selection_screen()



