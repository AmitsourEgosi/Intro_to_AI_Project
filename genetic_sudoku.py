import numpy as np
from functools import cmp_to_key
import config


def fitness(board):
    """
    fitness function that measures how fit is a given board in the sense of being close to
    the solution. we count amount of missing numbers in the rows, cols and boxes
    :param board: given board
    :return: fitness score. is negative as we try to maximize it to the optimum of 0
    """
    board = np.array(board)
    row_missing_nums = 9 - np.array([len(np.unique(board[i, :][board[i, :] != 0])) for i in range(9)])
    col_missing_nums = 9 - np.array([len(np.unique(board[:, j][board[:, j] != 0])) for j in range(9)])
    box_missing_nums = 9 - np.array([len(np.unique(board[i:i + 3, j:j + 3][board[i:i + 3, j:j + 3] != 0]))
                                     for i in range(0, 9, 3) for j in range(0, 9, 3)])
    return -(np.sum(row_missing_nums) + np.sum(col_missing_nums) + np.sum(box_missing_nums))


def fitness2(board):
    """
    another possible fitness function. checks sum, prod and existence of all numbers of rows cols and boxes
    :param board: given board
    :return: fitness score
    """
    board = np.array(board)
    row_sum = np.sum([np.abs(45 - np.sum(board[i, :])) for i in range(9)])
    col_sum = np.sum([np.abs(45 - np.sum(board[:, j])) for j in range(9)])
    box_sum = np.sum([np.abs(45 - np.sum(board[i:i + 3, j:j + 3])) for i in range(0, 9, 3) for j in range(0, 9, 3)])
    row_prod = np.sum(np.sqrt([np.abs(362880 - np.prod(board[i, :])) for i in range(9)]))
    col_prod = np.sum(np.sqrt([np.abs(362880 - np.prod(board[:, j])) for j in range(9)]))
    box_prod = np.sum(np.sqrt([np.abs(362880 - np.prod(board[i:i + 3, j:j + 3]))
                               for i in range(0, 9, 3) for j in range(0, 9, 3)]))
    row_set = np.sum([9 - len(np.intersect1d(board[i, :], range(1, 10))) for i in range(9)])
    col_set = np.sum([9 - len(np.intersect1d(board[:, j], range(1, 10))) for j in range(9)])
    box_set = np.sum([9 - len(np.intersect1d(board[i:i + 3, j:j + 3], range(1, 10)))
                      for i in range(0, 9, 3) for j in range(0, 9, 3)])
    return -(10 * (row_sum + col_sum + box_sum) + row_prod + col_prod + box_prod + 50 * (row_set + col_set + box_set))


class SudokuBoard:
    """
    wrapper of a board. an individual in our population
    """

    def __init__(self, fitness_fn, board):
        self.board = board
        self.fitness = fitness_fn(board)


class GeneticSolver:
    """
    class of a genetic algorithm to solve the sudoku puzzle. the population is a group of individuals,
    each individual is a sudoku bord
    """

    def __init__(self, population_size, elite_size, fitness_fn, selection_rate, mutation_prob, generation_num,
                 max_stuck):
        self.population_size = population_size
        self.elite_size = elite_size
        self.fitness_fn = fitness_fn
        self.selection_rate = selection_rate
        self.mutation_prob = mutation_prob
        self.generation_num = generation_num
        self.max_stuck = max_stuck
        self.valid_values = []

    def init_population(self, quiz, size):
        """
        creating a population of given size
        :param quiz: the given quiz to solve. all individuals do not violate sudoku rules with respect to the quiz
        :param size: size of population
        :return: population
        """
        po = set()
        while len(po) < size:
            new_in = np.zeros((9, 9))
            for i in range(9):
                for j in range(9):
                    if quiz[i, j] > 0:
                        new_in[i, j] = quiz[i, j]
                    else:
                        valid_values = self.get_valid_values(quiz, i, j)
                        new_in[i, j] = np.random.choice(valid_values)
            po.add(SudokuBoard(self.fitness_fn, new_in))
        return list(po)

    def get_valid_values(self, quiz, i, j):
        """
        return the valid values to place in (i,j) with respect to the quiz
        :param quiz: given quiz
        :param i: row
        :param j: col
        :return: list of values
        """
        return self.valid_values[i][j]

    def init_vv(self, quiz, i, j):
        # return np.arange(1,10)
        valid_values = []
        for number in range(1, 10):
            row, col = quiz[i, :], quiz[:, j]
            box = quiz[3 * (i // 3):3 * (i // 3) + 3, 3 * (j // 3):3 * (j // 3) + 3]
            if number not in row and number not in col and number not in box:
                valid_values.append(number)
        return valid_values

    def select_parent(self, population):
        """
        select a parent by conducting a fitness competition between to random individuals
        :param population: population to select from
        :return: parent
        """
        indices = np.random.choice(range(self.population_size), 2, replace=False)
        competitor1, competitor2 = population[indices[0]], population[indices[1]]
        fitter, weaker = competitor1, competitor2
        if competitor2.fitness > competitor1.fitness:
            fitter, weaker = competitor2, competitor1
        p = np.random.uniform()
        if p < self.selection_rate:
            return fitter
        else:
            return weaker

    def random_selection(self, population):
        """
        using natural random selection to choose two different parents
        :param population: population to choose from
        :return: two parents
        """
        parent1 = self.select_parent(population)
        parent2 = self.select_parent(population)
        while np.array_equal(parent2.board, parent1.board):
            parent2 = self.select_parent(population)
        return parent1, parent2

    def reproduce(self, parent1, parent2):
        """
        mate two parents using whole-row multi-point cross-over in order to create two children
        :return: two newborn children
        """
        number_of_swaps = np.random.choice(np.arange(1, 9))
        swapping_indices = np.random.choice(np.arange(0, 9), number_of_swaps, replace=False)
        child1, child2 = parent1.board.copy(), parent2.board.copy()
        for i in swapping_indices:
            child1[i, :] = parent2.board[i, :]
            child2[i, :] = parent1.board[i, :]
        return child1, child2

    def mutate(self, child, quiz):
        """
        swap mutation of non-fixed genes (numbers)
        :return: mutated child
        """
        mutated_child = np.copy(child)
        done = False
        while not done:
            row = np.random.randint(0, 9)
            source_col, dest_col = np.random.randint(0, 9), np.random.randint(0, 9)
            while source_col == dest_col:
                dest_col = np.random.randint(0, 9)
            not_fixed_condition = quiz[row, source_col] == 0 and quiz[row, dest_col] == 0
            valid_swap_values = child[row, source_col] in self.get_valid_values(quiz, row, dest_col) and \
                                child[row, dest_col] in self.get_valid_values(quiz, row, source_col)
            if not_fixed_condition and valid_swap_values:
                temp = child[row, dest_col]
                mutated_child[row, dest_col] = child[row, source_col]
                mutated_child[row, source_col] = temp
                done = True
        return mutated_child

    def sort_population(self, individual1, individual2):
        """
        comparator between to individuals (boards)
        :param individual1:
        :param individual2:
        :return:
        """
        if individual1.fitness < individual2.fitness:
            return -1
        elif individual1.fitness == individual2.fitness:
            return 0
        return 1

    def get_elite(self, population):
        """
        get the elite of the population
        :param population: given population
        :return: elite
        """
        elite = []
        for i in range(self.elite_size):
            elite.append(population[i])
        return elite

    def solve(self, quiz):
        """
        genetic algorithm to find maximum of the fitness function i.e a solution to the given sudoku quiz
        :param quiz: given sudoku quiz
        :return: solution if found during given number of generations.
        otherwise the best board at the last generation
        """
        self.valid_values = [[0] * 9 for i in range(9)]
        for i in range(9):
            for j in range(9):
                self.valid_values[i][j] = self.init_vv(quiz, i, j)

        population = sorted(self.init_population(quiz, self.population_size),
                            key=cmp_to_key(self.sort_population), reverse=True)
        best_individual = population[0]
        best_fitness = best_individual.fitness
        stale_count = 0
        for generation in range(self.generation_num):
            if best_fitness == 0:
                break
            if config.Debug:
                print("============== generation:", generation, " =================")
                print(best_fitness)
            new_population = []
            if stale_count == self.max_stuck:
                if config.Debug:
                    print("stuck! ; starting over...")
                population = sorted(self.init_population(quiz, self.population_size),
                                    key=cmp_to_key(self.sort_population), reverse=True)
            elite = self.get_elite(population)  # save elite
            for i in range(self.elite_size, self.population_size - 5, 2):  # new population
                parent1, parent2 = self.random_selection(population)
                child1, child2 = self.reproduce(parent1, parent2)
                prob1, prob2 = np.random.uniform(size=2)
                if prob1 > self.mutation_prob:
                    child1 = self.mutate(child1, quiz)
                if prob2 > self.mutation_prob:
                    child2 = self.mutate(child2, quiz)
                new_population.append(SudokuBoard(self.fitness_fn, child1))
                new_population.append(SudokuBoard(self.fitness_fn, child2))

            new_population += elite
            aliens = self.init_population(quiz, 5)
            new_population += aliens  # add 5 aliens to allow sudden variations
            population = sorted(new_population, key=cmp_to_key(self.sort_population), reverse=True)
            best_individual = population[0]
            if best_individual.fitness == best_fitness and best_individual.fitness == population[1].fitness:
                stale_count += 1
            else:
                stale_count = 0
            best_fitness = best_individual.fitness
        return best_individual
