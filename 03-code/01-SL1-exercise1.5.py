# Required imports
import numpy as np
import pandas as pd

import uuid
import random

# Sample games for testing

games = {
    "game1" : np.array([
        np.zeros((3,3), dtype=int),
        np.array([[1, 0, 0], 
                  [0, 0, 0], 
                  [0, 0, 0]]),
        np.array([[1, 0, 0], 
                  [0, -1, 0], 
                  [0, 0, 0]]),
        np.array([[1, 1, 0], 
                  [0, -1, 0], 
                  [0, 0, 0]]),
        np.array([[1, 1, -1], 
                  [0, -1, 0], 
                  [0, 0, 0]]),
        np.array([[1, 1, -1], 
                  [0, -1, 0], 
                  [1, 0, 0]]),
        np.array([[1, 1, -1], 
                  [0, -1, -1], 
                  [1, 0, 0]]),
        np.array([[1, 1, -1], 
                  [0, -1, -1], 
                  [1, 0, 1]]),
        np.array([[1, 1, -1], 
                  [-1, -1, -1], 
                  [1, 0, 1]])
    ]),
    "game2" : np.array([
        np.zeros((3,3), dtype=int),
        np.array([[0, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, 0]]),
        np.array([[0, 0, -1], 
                  [0, 1, 0], 
                  [0, 0, 0]]),
        np.array([[0, 0, -1], 
                  [0, 1, 0], 
                  [1, 0, 0]]),
        np.array([[0, 0, -1], 
                  [0, 1, -1], 
                  [1, 0, 0]]),
        np.array([[0, 0, -1], 
                  [0, 1, -1], 
                  [1, 0, 1]]),
        np.array([[0, 0, -1], 
                  [0, 1, -1], 
                  [1, -1, 1]]),
        np.array([[1, 0, -1], 
                  [0, 1, -1], 
                  [1, -1, 1]])
    ])
}

# Utility functions

def board_to_vector(board):
    assert board.shape == (3, 3)
    return board.reshape(-1)   # shape (9,)

def vector_to_board(vector):
    vector = np.asarray(vector)
    assert vector.shape == (9,)
    return vector.reshape(3, 3)

def get_available_moves(board):
    return list(zip(*np.where(board == 0)))

def get_threaths(board):
    threaths = np.concatenate([
        board.sum(axis=0),
        board.sum(axis=1),
        [board.diagonal().sum()],
        [np.fliplr(board).diagonal().sum()]
        ])
    return threaths

def get_winner(board):
    threaths = get_threaths(board)
    if 3 in threaths:
        return 1
    elif -3 in threaths:
        return -1
    elif 0 not in board:
        return 0
    else:
        return None
    
# Strategies
def strategy_double_attack(board, player=1):
    opponent = -player
    available_moves = get_available_moves(board)
    
    if not available_moves:
        return None

    # --- 1. WIN ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player
        if 3 * player in get_threaths(temp_board):
            return [move] 

    # --- 2. BLOCK ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = opponent
        if 3 * opponent in get_threaths(temp_board):
            return [move] 

    # --- 3. DOUBLE ATTACK (FORK) ---
    # Look for a move that creates 2 or more threats of "3"
    fork_moves = []
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player
        
        # Count how many lines now have a sum of 2*player 
        # (Meaning 2 of your pieces and 1 empty space)
        # Note: We check for 2*player because the move we just made 
        # creates a "threat" that the opponent must block.
        n_threats = np.sum(get_threaths(temp_board) == 2 * player)
        
        if n_threats >= 2:
            return [move]
    
    return available_moves

def strategy_center_block_random(board, player=1):
    opponent = -player
    available_moves = get_available_moves(board)
    
    if not available_moves:
        return None

    # --- 1. WIN: Can we win in this move? ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player
        if 3 * player in get_threaths(temp_board):
            return [move]

    # --- 2. BLOCK: Does the opponent have two in a row? ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = opponent # Simulate opponent playing there
        if 3 * opponent in get_threaths(temp_board):
            return [move] # Block that spot

    # --- 3. CENTER: Take the center if available ---
    center = (1, 1)
    if board[center] == 0:
        return [center]

    return available_moves

def strategy_corner_trap(board, player=1):
    opponent = -player
    available_moves = get_available_moves(board)
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    available_corners = [c for c in corners if c in available_moves]

    if not available_moves:
        return None
    
    # If it's the first move, take a corner
    if len(available_moves) >= 8 and available_corners:
        return [random.choice(corners)]

    # --- 1. WIN: Can we win in this move? ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player
        if 3 * player in get_threaths(temp_board):
            return [move]

    # --- 2. BLOCK: Does the opponent have two in a row? ---
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = opponent # Simulate opponent playing there
        if 3 * opponent in get_threaths(temp_board):
            return [move] # Block that spot

    # --- 3. CENTER: Take the center if available ---
    center = (1, 1)
    if board[center] == 0:
        return [center]

    return available_moves

def strategy_avoid_losing_moves(board, player=1):
    opponent = -player
    available_moves = get_available_moves(board)
    non_losing_moves = []
    
    # Case 1. Can we win this turn?
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player
        if 3 * player in get_threaths(temp_board):
            return [move]

    # Caso 2: Minimizar la máxima amenaza futura del oponente (Minimax Profundidad 2)
    min_max_threats = float('inf') 
    
    for move in available_moves:
        temp_board = board.copy()
        temp_board[move] = player # Simulamos nuestro movimiento
        
        opponent_moves = get_available_moves(temp_board)
        current_move_max_threats = 0
        
        for opp_move in opponent_moves:
            temp_board[opp_move] = opponent # Simulamos respuesta del oponente
            
            # Si con este movimiento el oponente GANA, es una amenaza infinita
            if 3 * opponent in get_threaths(temp_board):
                current_move_max_threats = 99 # Valor de castigo
                temp_board[opp_move] = 0 # LIMPIEZA
                break 
            
            # Si no gana, contamos cuántas amenazas de "fork" crea
            n_threats = np.sum(get_threaths(temp_board) == 2 * opponent)
            current_move_max_threats = max(current_move_max_threats, n_threats)
            
            temp_board[opp_move] = 0 # LIMPIEZA IMPORTANTE
        
        # Guardamos el movimiento que nos deje con el menor "daño máximo"
        if current_move_max_threats < min_max_threats:
            min_max_threats = current_move_max_threats
            non_losing_moves = [move]
        elif current_move_max_threats == min_max_threats:
            non_losing_moves.append(move)
            
    return non_losing_moves

class TicTacToe:
    def __init__(self, strategy_player1, strategy_player2):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        self.strategies = {1: strategy_player1, -1: strategy_player2}
        

    def __print__(self):
        symbol_map = {1: 'X', -1: 'O', 0: ' '}
        print('=' * 5)
        for row in self.board:
            print('|'.join(symbol_map[cell] for cell in row))
            print('-' * 5)
        print('=' * 5)

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player = - self.current_player  # Switch player
            return True
        return False
    
    def next_moves(self):
        strategy = self.strategies[self.current_player]
        return strategy(self.board, self.current_player)

    def check_winner(self):
        return get_winner(self.board)

    def reset(self):
        self.board.fill(0)
        self.current_player = 1

def play_game(game, strategy_player1, strategy_player2):
    game_id = str(uuid.uuid4())
    records = []
    for turn in range(9):
        board_before = game.board.copy()
        moves = game.next_moves()
        move = random.choice(moves)
        game.make_move(*move)
        winner = game.check_winner()
        records.append({
            "game_id": game_id,
            "strategy_p1": strategy_player1,
            "strategy_p2": strategy_player2,
            "turn": turn,
            "player": -game.current_player,
            "board": board_to_vector(board_before),
            "move": int(move[0] * 3 + move[1]),
            "winner": winner
        })
        if winner is not None:
            for r in records:
                r["winner"] = winner
            break
    return records
        
# Example of playing a game
if __name__ == "__main__":

    strategies = [
        strategy_double_attack,
        strategy_center_block_random,
        strategy_corner_trap,
        strategy_avoid_losing_moves
    ]
    dataset = []
    n_games_per_pair = 10
    for strategy_p1 in strategies:
        for strategy_p2 in strategies:
             for _ in range(5):  # Play multiple games for each pair of strategies
                game = TicTacToe(strategy_p1, strategy_p2)
                game_history = play_game(game, strategy_p1.__name__, strategy_p2.__name__)
                dataset.extend(game_history)
        
    # Save dataset to a DataFrame in parquet format
    df = pd.DataFrame(dataset)
    df.to_parquet("tictactoe_dataset.parquet", index=False)