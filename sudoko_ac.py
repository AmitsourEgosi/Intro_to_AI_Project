# Naive backtrack
def is_valid_assign(board, num, i, j):
    """
    Checks if num is valid assignment at (i,j) in relation to board
    :param board: the current given board
    :param num: the number to check
    :param i: the relevant row
    :param j: the relevant column
    :return: boolean (T/F)
    """
    sqr = int(len(board) ** 0.5)
    for row in range((i // sqr) * sqr, (i // sqr) * sqr + sqr):
        for col in range((j // sqr) * sqr, (j // sqr) * sqr + sqr):
            if board[row][col] == num:
                return False
    for piv in range(9):
        if board[i][piv] == num or board[piv][j] == num:
            return False
    return True


def next_by_order(board, csp=None):
    """
    Find the next empty cell on the board
    :param board: the current given board
    :return: the next empty cell row, the next empty cell column
    if there is no empty cell - it'll return -1,-1
    """
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return -1, -1


def mrv_heuristic(board, csp):
    """
    Minimum-remaining-valus heuristic, also has been called
    the “most constrained variable”
    :param board: The current board
    :param csp: The general csp problem
    :return: the next (in relate to mrv heuristic) empty cell
    """
    var = min(list(csp.dom.keys()), key=lambda x: len(csp.dom[x]) if len(csp.dom[x]) > 1 else 10)
    x, y = csp.chars[var[0]] - 1, int(var[1]) - 1
    if len(csp.dom[var]) == 1:
        return -1, -1
    return x, y


def degree_heuristic(board, csp):
    """
    The degree heuristic, return the cell which involves in
    largest largest number of constraints on other unassigned cells
    :param board: The current board
    :param csp: The general csp problem
    :return: the next (in relate to degree heuristic) empty cell
    """
    get_var_degree = lambda x: sum(
        [len(csp.dom[var]) if len(csp.dom[var]) > 1 else 0 for var in csp.neighbors[x]]) if len(
        csp.dom[x]) > 1 else 300
    var = min(list(csp.neighbors.keys()), key=get_var_degree)
    x, y = csp.chars[var[0]] - 1, int(var[1]) - 1
    if get_var_degree(var) == 300:
        return -1, -1
    return x, y


def LCV_heuristic(i, j, csp):
    """
    Least constraint value heuristic
    :param i: The row of the chosen variable
    :param j: The column of the chosen variable
    :param csp: The current csp problem
    :return: list of values from the (i,j) cell domain, sorted by LCV heuristic
    """
    get_lcv_val = lambda x: sum([1 if x in csp.dom[var] else 0 for var in csp.neighbors[csp.vars[i][j]]])
    res = sorted(list(csp.dom[csp.vars[i][j]]), key=get_lcv_val)
    return res


# Backtraking + arc consistency
def naive_pls_ac(board, var_heuristic=next_by_order, val_heuristic=lambda x, y, csp: list(csp.dom[csp.vars[x][y]])):
    """
    Backtrack function which involve different csp heuristics
    :param board: The current quiz board
    :param var_heuristic: The heuristic to choose the next variable
    :param val_heuristic: The heuristic to choose the next value to fix variable
    :return: the solution (if exists) or None
    """
    ac = AC_solver(board)
    if not ac.arc_consistency():
        return None
    board = ac.board
    i, j = var_heuristic(board, ac)
    if i == - 1 and j == - 1 and board[i][j] != 0:
        ac.update_board()
        return board
    val_list = val_heuristic(i, j, ac)
    for num in val_list:
        board[i][j] = num
        if naive_pls_ac(board, var_heuristic, val_heuristic) is not None:
            return board
        board[i][j] = 0
    return None


def naive_solve(board):
    """
    Backtrack function
    :param board: The current quiz board
    :return: the solution (if exists) or None
    """
    i, j = next_by_order(board)
    if i == - 1 and j == - 1 and board[i][j] != 0:
        return board
    for num in range(1, 10):
        if is_valid_assign(board, num, i, j):
            board[i][j] = num
            if naive_solve(board) is not None:
                return board
            board[i][j] = 0
    return None


# Arc consistency solver class
class AC_solver:
    def __init__(self, board):
        """
        Arc consistency constructor get board and initialize all the
        relevant parameters
        """
        self.chars = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9}
        self.nums = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.vars = [[char + num for num in self.nums] for char in self.chars]
        self.board = board
        self.arcs, self.neighbors = self.init_arcs()
        self.dom = self.init_domains()

    def get_neighbors(self, i, j):
        """
        This function initialize all the constrains which relevant
        to the (i,j) board cell
        :param i: the cell row
        :param j: the cell column
        :return: edges - set of all the (i,j) cell constrains edges,
                 neighbors - set of all the (i,j) cell neighbors
        """
        edges = set()
        neigbhors = set()
        sqr = int(len(self.board) ** 0.5)
        for row in range((i // sqr) * sqr, (i // sqr) * sqr + sqr):
            for col in range((j // sqr) * sqr, (j // sqr) * sqr + sqr):
                if (row, col) != (i, j):
                    edges.add((self.vars[i][j], self.vars[row][col]))
                    neigbhors.add(self.vars[row][col])
        for piv in range(len(self.board)):
            if piv != j:
                edges.add((self.vars[i][j], self.vars[i][piv]))
                neigbhors.add(self.vars[i][piv])
            if piv != i:
                edges.add((self.vars[i][j], self.vars[piv][j]))
                neigbhors.add(self.vars[piv][j])
        return edges, neigbhors

    def init_arcs(self):
        """
        This function initialize the arcs (= cells constrains) list
        and adjacency dictionary which match between cell (=key) and
        a set of all the cell which relevant to him
        """
        arcs = []
        neighbors = dict()
        for i in range(len(self.vars)):
            for j in range(len(self.vars)):
                new_arcs, cur_neighbors = self.get_neighbors(i, j)
                arcs += new_arcs
                neighbors[self.vars[i][j]] = cur_neighbors
        return arcs, neighbors

    def init_domains(self):
        """
        :return: An initialize dictionary with the initial domain (value)
        of any board cell (key)
        """
        dom = dict()
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] != 0:
                    dom[self.vars[i][j]] = {self.board[i][j]}
                else:
                    dom[self.vars[i][j]] = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        return dom

    def update_board(self):
        """
        This function updates the board in accordance to the
        cells domains after every iteration of the arc-consistency method
        """
        for i in range(len(self.vars)):
            for j in range(len(self.vars)):
                if len(self.dom[self.vars[i][j]]) == 1:
                    self.board[i][j] = list(self.dom[self.vars[i][j]])[0]

    def arc_consistency(self):
        """
        This function operate the correct elimination to ensure that
        after the function end so any variable (=board cell) in the problem
        will be arc-consistent/ he's domain will be empty
        :return: false if an inconsistency is found and true otherwise
        """
        while self.arcs:
            x_i, x_j = self.arcs.pop(0)
            if self.revised(x_i, x_j):
                if len(self.dom[x_i]) == 0:
                    return False
                for x_k in self.neighbors[x_i]:
                    if x_k != x_j:
                        self.arcs.append((x_k, x_i))

        return True

    def revised(self, x_i, x_j):
        """
        If the x_j domain is equals to {x} when x includes in
        the x_i domain - the function will delete x from x_i domain
        :param x_i: some board cell
        :param x_j: some board cell which has a constraint in relation
                    to x_i
        :return: True if we revised the x_i domain
        """
        revised = False
        for x in self.dom[x_i]:
            if {x} == self.dom[x_j]:
                val_to_del = x
                revised = True
                break
        if revised:
            self.dom[x_i].remove(val_to_del)
        return revised


def get_len(dict):
    res = 0
    for key in dict:
        res += len(dict[key])
    return res


def print_board(board):
    print("####################")
    for row in board:
        print(*row)


def solve_ac(board):
    global ac_solution
    ac = AC_solver(board)
    ac.arc_consistency()
    ac.update_board()
    return ac.board
