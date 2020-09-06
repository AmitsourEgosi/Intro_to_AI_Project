import numpy as np
import util
import copy


class EightPuzzle:
    def __init__(self, start_board):
        self.board = start_board
        self.size = len(self.board)
        self.goal_state = self.init_goal_state(len(self.board))
        self.expanded = 0

    def init_goal_state(self, len):
        res = []
        for row in range(self.size):
            cur_row = []
            for col in range(self.size):
                cur_row.append(col + 1 + row * len)
            res.append(cur_row)
        res[-1][-1] = -1
        return res

    def get_goal_state(self):
        return self.goal_state

    def get_start_state(self):
        return self.board

    def is_goal_state(self, state):
        return state == self.goal_state

    def get_successors(self, state):
        def switch_between(x1, y1, x2, y2):
            moc_state = copy.deepcopy(state)
            moc_state[x1][y1] = moc_state[x2][y2]
            moc_state[x2][y2] = -1
            return moc_state, 1

        self.expanded = self.expanded + 1
        x, y = np.where(np.array(state) == -1)
        x, y = x[0], y[0]
        res = []
        if x > 0:
            res.append(switch_between(x, y, x - 1, y))
        if x < self.size - 1:
            res.append(switch_between(x, y, x + 1, y))
        if y > 0:
            res.append(switch_between(x, y, x, y - 1))
        if y < self.size - 1:
            res.append(switch_between(x, y, x, y + 1))
        return res

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class ComparableState:
    """
    This class is a wrapper class to all the nodes.
    The class allows us to do comparison between the
    state nodes
    """

    def __init__(self, state, cur_value):
        self.state = state
        self.cur_value = cur_value

    def __iter__(self, other_state):
        return self.cur_value < other_state.eval_value

    def __gt__(self, other_state):
        return self.cur_value > other_state.cur_value


##############
# Heuristics #
##############
def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def misplaced_tiles(state, problem):
    x = problem.size ** 2 - (np.array(state) == np.array(problem.goal_state)).sum()
    return problem.size ** 2 - (np.array(state) == np.array(problem.goal_state)).sum()


def manhattan_dis(board, problem):
    size = len(board)
    res = 0
    for i in range(size):
        for j in range(size):
            cur_val = board[i][j] - 1
            if cur_val == -2:
                cur_val = size * size - 1
            res += abs(i - cur_val // size) + abs(j - cur_val % size)
    return res


def by_order(board, problem):
    size = len(board)
    res = 0
    for i in range(len(board)):
        for j in range(len(board)):
            cur_val = board[i][j] - 1
            if cur_val == -2:
                cur_val = size * size - 1
            # add the (manhattan distance)*(the cell priority value)
            res += (1 - cur_val / (size ** 2 - 1)) * (abs(i - cur_val // size) + abs(j - cur_val % size))
            # res += ((size**2 - 1) - cur_val)*(abs(i - cur_val // size) + abs(j - cur_val % size))
    return res


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    explored = set()
    fringe = util.PriorityQueue()
    root = ComparableState((problem.get_start_state()), heuristic(problem.get_start_state(), problem))
    fringe.push((root, [problem.get_start_state()]),
                heuristic(problem.get_start_state(), problem))  # time complexity of O(1)?
    while not fringe.isEmpty():  # time complexity of O(1)
        (current, actions) = fringe.pop()  # time complexity of log(n)
        state_key = str(current.state)
        if state_key not in explored:  # time complexity of O(1)
            # print_state(current.state)
            if problem.is_goal_state(current.state):  # time complexity of O(1)
                return actions
            explored.add(state_key)  # time complexity of O(1)
            curr_cost = problem.get_cost_of_actions(actions)  # O(len(actions))
            for child in problem.get_successors(current.state):
                new_eval_val = curr_cost + problem.get_cost_of_actions([child[1]]) + heuristic(child[0],
                                                                                               problem)
                fringe.push((ComparableState(child[0], new_eval_val), actions + [child[0]]),
                            new_eval_val)  # time complexity of log(n)
    return None


def print_state(state):
    for row in state:
        print('[', end='')
        for col in row:
            if col != row[-1]:
                print(col, ' ', end='')
            else:
                print(col, end='')
        print(']')
    print("*******")


def print_path(path_to_solve, problem):
    if not path_to_solve:
        print("could not solve the puzzle")
        return
    for state in path_to_solve:
        print_state(state)
    print("length: ", len(path_to_solve))
    print("expanded: ", problem.expanded)


def solve_arr(puzzle):
    puzzle[puzzle == 0] = -1
    problem = EightPuzzle(puzzle.tolist())
    path_to_solve = a_star_search(problem, manhattan_dis)
    print_path(path_to_solve, problem)
    return True


