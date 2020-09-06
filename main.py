import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import read_image
import sudoku_dlx
import sudoko_ac, sudoku_pencil, sudoku_simulated_annealing
import genetic_sudoku as sudoku_genetic
import neural_net_model as nnm
import data_loader as dl
import config
import eight_puzzle


def train():
    """train the nn"""
    training_data, test_data = dl.load_printed_data()
    nn = nnm.NN([784, 128, 30, 10], nnm.CrossEntropyLoss, nnm.ReLU, nnm.Sigmoid)
    nn.train(training_data + test_data, num_of_epochs=5, mini_batch_size=10, reg=5.0, eta=0.5)
    # print("accuracy: ", nn.score(test_data))
    nnm.save_to_pkl(nn, 'neural_net_printed_8.pickle')


def get_board_from_image(f, nn, puzzle_size):
    if not f:
        f = random_image()
    digits, im = read_image.extract_puzzle(f, puzzle_size)
    puzzle = read_image.read_digits(digits, nn=nn)
    return puzzle, im


def random_image():
    d = "sample_images"
    r1 = random.Random()
    f = r1.choice(os.listdir(d))  # change dir name to whatever
    print(f"random image selected ={f}")
    f = os.path.join(d, f)
    return f


def test_algo_pd(algo, n, difficulty=True, data2=False):
    if difficulty == "data2":
        puzzles, s = load_data2(n)
        t = test_algo2(algo, puzzles, s)

        print(f"data 2 on {n} puzzles, {t / n}")
        return

    puzzles = load_data()
    if isinstance(difficulty, str):
        p = puzzles.loc[puzzles.Difficulty == difficulty].head(n)
        t = test_algo(algo, p)
        print(f"{difficulty}, {t / len(p.index)}")


    elif difficulty:
        for d in puzzles.Difficulty.unique():
            p = puzzles.loc[puzzles.Difficulty == d].head(n)
            t = test_algo(algo, p)
            print(f"{d}, {t / len(p.index)}")

    else:
        t = test_algo(algo, puzzles.head(n))
        print(f"all - time taken: {t}")


def test_algo(algo, puzzles):
    """
    test the algorithm on pandas data set
    :return: time taken
    """
    t, f = 0, 0
    start = time.time()
    for index, p in puzzles.iterrows():
        sol = algo(p.Puzzle)
        if np.all(sol == p.Solution):
            t += 1
            if config.Debug:
                print('t', end=' ')
        else:
            f += 1
            if config.Debug:
                print('f', end=' ')
        if config.Debug:
            print(index)
    if config.Debug:
        print(f"t={t},f={f}")
    end = time.time()
    return end - start


def test_algo2(algo, puzzles, solution):
    """
    test the algorithm on numpy data set
    :return: time taken
    """
    t, f = 0, 0
    start = time.time()
    for index, (p, s) in enumerate(zip(puzzles, solution)):
        sol = algo(p)
        if np.all(sol == s):
            t += 1
            if config.Debug:
                print('t', end=' ')
        else:
            f += 1
            if config.Debug:
                print('f', end=' ')
        if config.Debug:
            print(index)
    if config.Debug:
        print(f"t={t},f={f}")
    end = time.time()
    return end - start


def load_data():
    """
    :param num_of_quizzes: MUST be less then or equal to 1000000
    :return:
    """
    df = pd.read_csv('sudoku2.csv', index_col=False)
    df['Puzzle'] = df.Puzzle.apply(lambda x: np.array([int(s) for s in x]).reshape(9, 9))
    df['Solution'] = df.Solution.apply(lambda x: np.array([int(s) for s in x]).reshape(9, 9))
    return df


def load_data2(num_of_quizzes):
    """loads data from the million sudoku"""
    quizzes = np.zeros((num_of_quizzes, 81), np.int32)
    solutions = np.zeros((num_of_quizzes, 81), np.int32)
    k = 0

    for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
        if k >= num_of_quizzes:
            break
        quiz, solution = line.split(",")
        for j, q_s in enumerate(zip(quiz, solution)):
            q, s = q_s
            quizzes[i, j] = q
            solutions[i, j] = s
        k += 1
    p = quizzes.reshape((-1, 9, 9))
    s = solutions.reshape((-1, 9, 9))
    return p, s


def readCommand(argv):
    """
    Processes the command used to run pacman from the command line.
    """

    def default(str):
        return str + ' [Default: %default]'

    from optparse import OptionParser
    usageStr = """
  USAGE: Please enter an image path using the -i option or random using -r 
  EXAMPLES:
    -r
    -i "sample_images/img2.jpg" -a pen
    -r -a ac_bc --hr MRV
    -i "sample_images_eight/img1.png" -a eight
  """
    parser = OptionParser(usageStr)

    parser.add_option('-n', '--numGames', dest='numGames', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=100)
    parser.add_option('-a', '--algo', dest='algo',
                      help=default('the algorithm to use'),
                      metavar='ALGO', default='dlx')
    parser.add_option('-i', '--image', dest='image_path',
                      help=default('the image to extract puzzle from'),
                      default=None)
    parser.add_option('--nn', dest='nn',
                      help=default('the neural net model pickle file to use'),
                      default='neural_net_printed_8.pickle')
    parser.add_option('-r', '--random_image', dest='random_image',
                      help=default('the image to extract puzzle from'),
                      default=False, action="store_true")
    parser.add_option('-s', '--test', dest='test',
                      help=default('run on test mode, measuring algorithm times'),
                      default=False, action="store_true")
    parser.add_option('-d', '--debug', dest='debug',
                      help=default('debug level'),
                      default=10)
    parser.add_option('--difficulty', dest='difficulty',
                      help=default('difficulty to use'),
                      default=5)
    parser.add_option('--pd', dest='per_difficulty',
                      help=default('per difficulty test'),
                      default=True)
    parser.add_option('--hr', '--heuristic', dest='heuristic',
                      help=default('heuristic to use when applicable'),
                      default=None)

    options, otherjunk = parser.parse_args(argv)

    if not options.test and not options.image_path and not options.random_image:
        raise Exception('Please enter an image path using the -i option or random using -r')

    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    algos = get_algos()
    algo_txt = options.algo.upper()
    options.algo_txt = algo_txt
    if algo_txt not in algos:
        raise Exception(f"algorithm {options.algo} not found. options are:\n"
                        f"{str(algos.keys())}")
    options.algo = algos[algo_txt]

    if algo_txt == 'AC':
        if options.heuristic:
            options.heuristic = options.heuristic.upper()
            if options.heuristic == 'MRV':
                options.algo = lambda x: sudoko_ac.naive_pls_ac(x, sudoko_ac.mrv_heuristic)
            elif options.heuristic == 'DH':
                options.algo = lambda x: sudoko_ac.naive_pls_ac(x, sudoko_ac.degree_heuristic)
            elif options.heuristic == 'DH_LCV':
                options.algo = lambda x: sudoko_ac.naive_pls_ac(x, sudoko_ac.degree_heuristic)
            else:
                raise Exception(f"heuristic {options.heuristic} not found. options are:\n"
                                f"MRV,DH,DH_LCV")

    if algo_txt == 'PEN':
        if options.heuristic not in sudoku_pencil.h:
            raise Exception(f"heuristic {options.heuristic} not found. options are:\n"
                            f"{sudoku_pencil.h}")

        sudoku_pencil.hur = options.heuristic

    return options


def get_algos():
    global PEN
    DLX = sudoku_dlx.solve
    PEN = sudoku_pencil.solve
    DFS = sudoko_ac.naive_solve
    S_A = lambda x: sudoku_simulated_annealing.solve(x, itr_limit=250000, T=0.8)
    AC = sudoko_ac.naive_pls_ac
    AC2 = sudoko_ac.solve_ac
    eight = eight_puzzle.solve_arr

    def GENETIC(x):
        solver = sudoku_genetic.GeneticSolver(1000, 50, sudoku_genetic.fitness, 0.85, 0.06, 1000, 20)
        return solver.solve(x).board

    return {"DLX": DLX,
            "PEN": PEN,
            "DFS": DFS,
            "S_A": S_A,
            "AC_BC": AC,
            "AC": AC2,
            "EIGHT": eight,
            "GENETIC": GENETIC}


def main():
    argv_ = sys.argv[1:]
    args = readCommand(argv_)  # Get game components based on input
    get_algos()
    from_image = not args.test
    image_path = args.image_path
    algo = args.algo
    config.Debug = int(args.debug) < 2
    np.random.seed(0)
    random.seed(0)
    eightPuzzle = args.algo_txt == "EIGHT"
    puzzle_size = 3 if eightPuzzle else 9
    if from_image:
        read_image.__level = int(args.debug)

        board, im = get_board_from_image(image_path, nn=args.nn, puzzle_size=puzzle_size)
        if args.random_image:
            read_image.imdisplay(im[0], 11, False, "Puzzle")

        sudoku_pencil.print_board(board)
        sol = algo(board)
        if eightPuzzle:
            return
        sudoku_pencil.print_board(sol)
        if sol is not None:
            solution = read_image.write_solution(*im, board, sol)
            read_image.imdisplay(solution, 11, False, "Solution")
            read_image.plt.show()
        else:
            print("Failed to solve the puzzle :(")


    else:
        test_algo_pd(algo, args.numGames, args.per_difficulty)


if __name__ == '__main__':
    # train()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
