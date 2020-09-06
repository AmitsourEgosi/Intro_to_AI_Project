
EMPTY_CELL = -1

hur = "FE"
h = {
    "FE",
    "LO",
    "LO_MAX",
    "LO_MIN"
}


class Sudoku:
    """
    the board
    """

    def __init__(self, board):
        """
        creates the board object, starting with creating the areas and then
        builds an empty board. last places each number in his place.
        :param board:
        """
        global hur
        h = {
            "FE": self.hur_first_empty,
            "LO": self.hur_least_options,
            "LO_MAX": self.hur_least_options_max_empty_neighbors,
            "LO_MIN": self.hur_least_options_min_empty_neighbors,
        }

        self.best_guess = h[hur]

        self.n = 9
        n = self.n
        self.n_sq = int(self.n ** 0.5)
        self.full_cell_options = [[i == j for i in range(n)] for j in range(n)]
        self.small_sqrs = [area(n) for i in range(n)]
        self.columns = [area(n) for i in range(n)]
        self.rows = [area(n) for i in range(n)]
        self.board = self.generate_empty_board()
        self.no_changes_flag = False
        self.areas_types = [self.rows, self.columns, self.small_sqrs]
        self.full_cells = 0
        # to undo changes keep the order
        self.filled_stack = list()
        # is the board valid, whe a method finds that the board cannot be solved
        # mark as False
        self.valid = True

        self.fill_board(board)

    def generate_empty_board(self):
        """
        generates the cells and connects them to the areas
        :return:
        """

        board = [[None] * self.n for i in range(self.n)]

        for r in range(self.n):
            for c in range(self.n):
                areas = [self.rows[r], self.columns[c],
                         self.get_small_sqr(r, c)]
                board[r][c] = cell(self, areas)
                board[r][c].r = r
                board[r][c].c = c
        return board

    def get_small_sqr(self, r, c):
        """
        finds the maching small squere to the cell
        :param r:
        :param c:
        :return:
        """
        return self.small_sqrs[r // self.n_sq + c - c % self.n_sq]

    def fill_board(self, board):
        """ sets the value of each cell to the input '-1' is becasue it is
        easyier to work with empty cell as -1"""

        for r in range(self.n):
            for c in range(self.n):
                if board[r][c] - 1 != EMPTY_CELL:
                    self.board[r][c].value = board[r][c] - 1

    def find_one_option_cells(self):
        """
        tells each cell to set its value to the only option if only one
        option is left
        :return:
        """
        for r in range(self.n):
            for c in range(self.n):
                if not self.valid:
                    return
                self.board[r][c].fill_if_one_option()

    def find_one_place_in_area(self, area_type):
        """
        for each area finds the number of options to place a number,
        if there are None the board is invalid, if the is only one then it
        is a sure thing
        :param area_type:
        :return:
        """
        for area_i in area_type:
            for number_option in range(self.n):
                if not area_i.valid_options[number_option]:
                    continue
                count_possible = 0
                found = None
                for cell in area_i.cells:
                    if cell.is_valid_option(number_option):
                        count_possible += 1
                        found = cell

                if not count_possible > 0:
                    self.valid = False
                    return

                if count_possible == 1:
                    found.value = number_option

    def find_sure_matches(self):
        """
        iterates on the sure matches, each cell if changed will call back to
        the sudoku object to sat the flag to false.
        :return:
        """
        while True:
            self.no_changes_flag = True
            self.find_one_option_cells()
            if self.is_full():
                return True
            for area_type in self.areas_types:
                self.find_one_place_in_area(area_type)
                if self.is_full():
                    return True
                if not self.valid:
                    return False
            if self.no_changes_flag:
                return True

    def __str__(self):
        """
        for debug
        :return:
        """
        return "[" + "\n,".join(str(r) for r in self.board) + \
               "]\n\n" + "█" * (self.n * 2 + 2) + "\n"

    def print_oprions(self):
        """
        for debug
        :return:
        """
        for r in range(self.n):
            for c in range(self.n):
                print((self.board[r][c].print_options()), end="")
            print()

    def is_full(self):
        """
        are alll the cells full?
        :return:
        """
        return self.full_cells == self.n ** 2

    def hur_least_options_max_empty_neighbors(self):
        min_options = self.n + 1
        max_effected = 0
        min_cell = None
        for cell_i in self:
            if cell_i.is_empty:
                if cell_i.valid_options_count <= min_options:
                    effected = sum([area_i.full_cells for area_i in
                                    cell_i.areas])
                    if cell_i.valid_options_count < min_options:
                        max_effected = effected
                        min_cell = cell_i
                    elif effected < max_effected:
                        max_effected = effected
                        min_cell = cell_i
        if min_cell is None:
            self.valid = False
            # raise InvalidBoardExeption()
        return min_cell

    def hur_least_options_min_empty_neighbors(self):
        min_options = self.n + 1
        min_effected = self.n * 3 + 1
        min_cell = None
        for cell_i in self:
            if cell_i.is_empty:
                if cell_i.valid_options_count <= min_options:
                    effected = sum([area_i.full_cells for area_i in
                                    cell_i.areas])
                    if cell_i.valid_options_count < min_options:
                        min_effected = effected
                        min_cell = cell_i
                    elif effected < min_effected:
                        min_effected = effected
                        min_cell = cell_i

    def hur_least_options(self):
        min_options = self.n + 1
        min_cell = None
        for cell_i in self:
            if cell_i.is_empty:
                if cell_i.valid_options_count < min_options:
                    min_cell = cell_i
                    min_options = cell_i.valid_options_count
                    if min_options == 2:
                        break

        if min_cell is None:
            self.valid = False
            # raise InvalidBoardExeption()
        return min_cell

    def hur_first_empty(self):
        min_options = self.n + 1
        min_cell = None
        for cell_i in self:
            if cell_i.is_empty:
                return cell_i

    def __iter__(self):
        """
        iter over the cells in the board
        :return:
        """
        for r in range(self.n):
            for c in range(self.n):
                yield self.board[r][c]

    def to_list(self, board=None):
        """
        reverse for fill board, fills the given list with the board object,
        +1 is because I used -1 as empty and the input used 0
        :param board:
        :return:
        """
        if board is None:
            board = [[0] * self.n for i in range(self.n)]
        for r in range(self.n):
            for c in range(self.n):
                board[r][c] = self.board[r][c].value + 1
        return board

    def remove_last(self):
        cell = self.filled_stack.pop(-1)
        cell.remove_value()


class area:
    """
    for example row, column or small square in order to reuse code
    it has link to its cells and counts the full cells and
    the valid_options that are left
    """

    def __init__(self, size):
        self.cells = []
        self.full_cells = 0
        self.size = size
        self.valid_options = [True] * size

    def update_options(self, filled_cell):
        """
        call back from a cell that was filled, tell all cells in the area
        to remove the option
        :param filled_cell:
        :return:
        """
        value = filled_cell.value
        self.full_cells += 1

        self.valid_options[value] = False

        for cell in self.cells:
            if cell is filled_cell:
                continue
            cell.remove_option(value)

    def return_option(self, emptyed_cell, value):
        """
        undo for update options, the cells value was emptyed
        :param emptyed_cell:
        :param value:
        :return:
        """
        if self.valid_options[value]:
            pass
            # raise Exception

        self.full_cells -= 1

        self.valid_options[value] = True

        for cell in self.cells:
            if cell is emptyed_cell:
                continue
            cell.return_option(value)


class cell:
    def __init__(self, soduko, areas):
        """
        the cell saves its value and a list of possible values, in order to
        support undo _valid_options[i] can be 1 = good or less becasue the
        option was removed in additon there are links to the areas and the
        board in order to call back after changing value
        :param soduko:
        :param areas:
        """
        self.areas = areas
        for area_i in areas:
            area_i.cells.append(self)
        self.soduko = soduko
        self._val = EMPTY_CELL
        self.valid_options_count = self.soduko.n
        self._valid_options = [1] * self.soduko.n

    @property
    def is_empty(self):
        return self._val == EMPTY_CELL

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, value):
        """
        after the value was set "callback" to the board and update its state
        and tell the area that the value was set
        :param value:
        :return:
        """
        if not self.is_empty:
            self.soduko.valid = False
            return
        if not self.is_valid_option(value):
            self.soduko.valid = False
            return
        self._val = value
        # self._valid_options = self.soduko.full_cell_options[value]
        # self.valid_options_count = 1
        self.soduko.no_changes_flag = False
        self.soduko.full_cells += 1
        self.soduko.filled_stack.append(self)
        for area_i in self.areas:
            area_i.update_options(self)

    def remove_value(self):
        """
        undo for set value
        :return:
        """
        value = self._val
        self._val = EMPTY_CELL
        self.soduko.full_cells -= 1
        for area_i in self.areas:
            area_i.return_option(self, value)

    def remove_option(self, value):
        """
        removes the option from the valid option by reducing it by 1
        :param value:
        :return:
        """
        if self._val == value:
            self.soduko.valid = False
            # raise InvalidBoardExeption
        self._valid_options[value] -= 1
        if self._valid_options[value] == 0:
            self.valid_options_count -= 1
        if self.valid_options_count <= 0:
            self.soduko.valid = False
            # raise InvalidBoardExeption

    def return_option(self, value):
        """
        adds one back to the option, if it is valid again add one to the
        counter
        :param value:
        :return:
        """

        self._valid_options[value] += 1
        if self._valid_options[value] == 1:
            self.valid_options_count += 1

    def is_valid_option(self, value):
        """is it my value or a valid option to put"""
        if self.is_empty:
            return self._valid_options[value] == 1
        return self.value == value

    def fill_if_one_option(self):
        """
        if there is one option left set it as my value
        :return:
        """
        if self.valid_options_count != self._valid_options.count(1):
            self.soduko.valid = False
            # raise InvalidBoardExeption
        if self.is_empty and self._valid_options.count(1) == 1:
            self.value = self.first_valid()

    def guess(self):
        """place the first valid in me"""
        self.value = self.first_valid()

    def first_valid(self):
        """the first valid option"""
        return next(i for i in range(self.soduko.n) if
                    self.is_valid_option(i))

    def get_possible_guesses(self):
        """all possible options"""
        return [i for i in range(self.soduko.n) if
                self.is_valid_option(i)]

    def __str__(self):
        """
        for debug
        :return:
        """
        return str(self.value + 1)

    def __repr__(self):
        """
        for debug
        :return:
        """
        return self.__str__()

    def print_options(self):
        """
        for debug
        :return:
        """
        s = "["

        for i in range(self.soduko.n):
            if self.is_valid_option(i):
                s += str(i + 1)
            else:
                s += "_"

        s += "]"
        return s


def solve(board):
    sudoku_obj = Sudoku(board)

    result, sudoku_obj = solve_sudoku_helper(sudoku_obj)
    if result:
        return sudoku_obj.to_list()


def solve_sudoku_helper(sudoku_obj):
    """
    recursive function find all sure matches, if the board is invalid return False
    if the board is full return true

    else make a guess:
    remember how many full cells and if the guess was wrong revert the board.
    guess in the best cell to guess in
    :param sudoku_obj:
    :return:
    """
    sudoku_obj.find_sure_matches()
    if not sudoku_obj.valid:
        return False, sudoku_obj
    full = sudoku_obj.is_full()
    # print(sudoku_obj)
    full_cells = sudoku_obj.full_cells

    if not full:
        # sudoku_obj.print_oprions()
        cell_i = sudoku_obj.best_guess()
        if cell_i is None:
            return False, sudoku_obj
        r = cell_i.r
        c = cell_i.c
        guesses = cell_i.get_possible_guesses()
        for i in guesses:
            result = True
            sudoku_obj.board[r][c].value = i

            result, sudoku_solution = solve_sudoku_helper(sudoku_obj)
            if result:
                return True, sudoku_solution

            # if the guessed board was unsolvable we guessed wrong
            # backtrack
            while sudoku_obj.full_cells > full_cells:
                sudoku_obj.remove_last()
            sudoku_obj.valid = True

        return False, sudoku_obj

    return True, sudoku_obj


def print_board(board):
    """
    for debug
    :param board:
    :return:
    """
    if board is None:
        print("None")
        return
    pretty_print = True

    n = len(board)
    ender = "," if pretty_print else ""
    for r in range(n):
        if pretty_print and r % n ** 0.5 == 0:
            print("-" * int(2.5 * n))
        for c in range(n):
            if pretty_print and c % n ** 0.5 == 0:
                print("|", end=",")
            print(board[r][c], end=ender)
        if pretty_print:
            print()

    print()


def print_rows(board):
    """
    for debug
    :param board:
    :return:
    """
    for row in board:
        print(row)
