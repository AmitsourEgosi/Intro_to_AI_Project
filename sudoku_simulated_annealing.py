import numpy as np
import random
from copy import deepcopy
from math import exp

DEBUG = "False"


class SimulatedAnnealing():
    def __init__(self, board, original_indices=None):
        # flatten the array in case it is 2d
        self.board = np.array(board).flatten()
        # save the indices of the original Sudoku entries:
        if original_indices is None:
            self.original_indices = [i for i in range(81) if self.board[i] != 0]
        else:
            self.original_indices = original_indices

    def get_original_indices(self):
        """
        Returns the indices of the original (given) entries of the Sudoku puzzle.
        """
        return self.original_indices

    def get_block_indices(self, block_num):
        """
        Returns the indices for a specific board block. there are 9 blocks in total, counting from the
        top left block (#1) to the bottom right (#9).
        """
        indices = []
        row_offset = (block_num // 3) * 3
        col_offset = (block_num % 3) * 3
        for i in range(9):
            indices.append(col_offset + (i % 3) + 9 * (row_offset + (i // 3)))
        return indices

    def get_column_indices(self, col_num):
        """
        Returns the indices for a given column from 1 to 9 (counting from left to right).
        """
        indices = [col_num + 9 * i for i in range(9)]
        return indices

    def get_row_indices(self, row_num):
        """
        Returns the indices for a given row from 1 to 9 (counting from top to bottom).
        """
        indices = [i + 9 * row_num for i in range(9)]
        return indices

    def get_score(self):
        """
        Returns a score between 0 and -243.
        for each unique entry in a row or column we subtract 1 from the score.
        """
        score = 0
        tmp_set = set()
        for col in range(9):
            for i in self.get_column_indices(col):
                tmp_set.add(self.board[i])
            score -= len(tmp_set)
            tmp_set.clear()

        for row in range(9):
            for j in self.get_row_indices(row):
                tmp_set.add(self.board[j])
            score -= len(tmp_set)
            tmp_set.clear()

        for block in range(9):
            for k in self.get_block_indices(block):
                tmp_set.add(self.board[k])
            score -= len(tmp_set)
            tmp_set.clear()

        return score

    def randomly_fill_blanks(self):
        """
        Randomly filling each 3x3 block.
        """
        for i in range(9):
            block_indices = self.get_block_indices(i)
            block = self.board[block_indices]
            zero_indices = [ind for i, ind in enumerate(block_indices) if block[i] == 0]
            to_fill = [i for i in range(1, 10) if i not in block]
            random.shuffle(to_fill)
            for ind, value in zip(zero_indices, to_fill):
                self.board[ind] = value

    def get_random_successor(self):
        """
        Generates a random successor by choosing a 2 blank squares at random, and filling
        them with a random value between 1-9.
        """
        new_board = deepcopy(self.board)
        j = 0
        k = 0
        while (j in self.original_indices):
            j = random.randint(0, 80)
        while (k in self.original_indices) or (k == j):
            k = random.randint(0, 80)
        new_board[j] = random.randint(1, 9)
        new_board[k] = random.randint(1, 9)

        return new_board

    def get_random_successor2(self):
        """
        Generates a random successor by choosing a 3x3 block at random, and swapping
        two entries within that square.
        """
        new_board = deepcopy(self.board)
        block = random.randint(0, 8)
        block_indices = self.get_block_indices(block)
        # delete original entry' indices:
        for i in self.original_indices:
            if i in block_indices:
                block_indices.remove(i)
        num_in_block = len(block_indices)
        random_squares = random.sample(range(num_in_block), 2)
        square1, square2 = [block_indices[ind] for ind in
                            random_squares]
        new_board[square1], new_board[square2] = new_board[square2], new_board[square1]

        return new_board

    def get_board(self):
        """
        Returns a 2d numpy array that is a Sudoku board.
        """
        board = np.reshape(self.board, (9, 9))
        return board


def get_temperature(curr_temp, cooling_schedule):
    if (cooling_schedule == "exp"):
        return curr_temp * 0.99999
    elif (cooling_schedule == "linear"):
        return curr_temp - 0.0000029


def solve(board, cooling_schedule="exp", T=0.4, itr_limit=150000):
    """
    Solves a Sudoku board using simulated annealing.
    """
    itr_counter = 0
    curr_board = SimulatedAnnealing(board)
    curr_board.randomly_fill_blanks()
    curr_score = curr_board.get_score()
    best_score = curr_score

    while itr_counter < itr_limit:
        itr_counter += 1

        if DEBUG:
            if itr_counter % 1000 == 0:
                print("iteration %s. best_score: %s. temp: %s" % (itr_counter, best_score, T))

        if itr_counter > 0.5 * itr_limit:
            next_board = SimulatedAnnealing(curr_board.get_random_successor(),
                                            curr_board.get_original_indices())
        else:
            next_board = SimulatedAnnealing(curr_board.get_random_successor2(),
                                            curr_board.get_original_indices())
        next_score = next_board.get_score()
        delta_score = float(curr_score - next_score)

        if next_score < curr_score:
            curr_board = next_board
            curr_score = next_score

        elif exp((delta_score / T)) - random.random() > 0:
            curr_board = next_board
            curr_score = next_score

        if (curr_score < best_score):
            best_score = curr_score

        if next_score == -243:
            curr_board = next_board
            best_score = next_score
            break

        T = get_temperature(T, cooling_schedule)

    if best_score == -243:
        return curr_board.get_board()
    else:
        return None
