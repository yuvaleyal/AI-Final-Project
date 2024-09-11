BOARD_SIZE = 8
BLACK = 1
BLACK_QUEEN = 10
WHITE = -1
WHITE_QUEEN = -10
QUEEN_MULTIPLIER = 10
EMPTY = 0
TIE = 0
NOT_OVER_YET = 2
CMD = [False]
OBJECTS_DIR = 'objects'
Q_Learning_OB_PATH = lambda player: f'{OBJECTS_DIR}/{player}_Q_LearningAgent.pkl'
AlphaZeroNET_OB_PATH = lambda player: f'{OBJECTS_DIR}/{player}_AlphaZeroNET.pkl'
MCTS_OB_PATH = lambda player: f'{OBJECTS_DIR}/{player}_MCTS.pkl'

PLAYER_NAME_A = "pa"
PLAYER_NAME_B = "pb"
HUMAN = 'human'
AlphaZero = 'alphazero'

EVAL_MODE = "eval"
TRAINING_MODE = "train"
TESTING_MODE = "test"
MODE = [EVAL_MODE]
PROG_MODE = [TRAINING_MODE, EVAL_MODE, TESTING_MODE]
TYPE_PLAYERS = ['random', HUMAN, 'minimax', 'rl', 'dnn', 'first_choice', AlphaZero]
ADVANCED_PLAYERS = []
