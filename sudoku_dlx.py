"""written by Iddo Ziv a readable oop implementation of algorithm X"""

import copy

from sudoku_pencil import *

import config


class DoubleNode:
    """
    double node, allowing some broken chains
    """
    after: None  # type: DoubleNode
    before: None  # type: DoubleNode

    def __init__(self, value=None, before=None, after=None):
        """
        generate a new node
        :param value:
        :param before:
        :param after:
        """
        self.value = value
        self.before = before
        self.after = after

    def __iter__(self):
        next_node = self
        while next_node.after is not None:
            next_node = next_node.after
            yield next_node

    def add_after(self, value):
        """
        add new node after self
        :rtype: DoubleNode
        :param value:
        :return:
        """
        next_node = DoubleNode(value, self, self.after)
        self.after = next_node
        if next_node.after is not None:
            next_node.after.before = next_node
        return self.after

    def re_attach(self):
        """
        reverse the pop
        :return:
        """
        if self.before is not None:
            self.before.after = self
        if self.after is not None:
            self.after.before = self

    def pop(self):
        """
        remove the node from the list
        :return:
        """
        if self.before is not None:
            self.before.after = self.after
        if self.after is not None:
            self.after.before = self.before

    @staticmethod
    def build_list(list):
        """
        convert list to linked list
        :param list:
        :return:
        """
        node_start = DoubleNode(None)
        if (len(list) == 0):
            return node_start
        node = node_start
        for item in list:
            node = node.add_after(item)
        node = node_start.after
        node.before = None
        return node

    def __str__(self):
        str_1 = str(self.value)
        if self.before is not None:
            str_1 = "<" + str_1
        if self.after is not None:
            str_1 = str_1 + ">"
        return "[" + str_1 + "]"

    def __repr__(self):
        return self.__str__()


class QuadNode:
    """
    double node, allowing some broken chains
    """
    right: None  # type: QuadNode
    left: None  # type: QuadNode
    up: None  # type: QuadNode
    down: None  # type: QuadNode
    text: None

    def __init__(self, ):
        """
        generate a new node
        :param header:
        :param left:
        :param right:
        """
        self.header = None
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.text = None

    def __iter__(self):
        next_node = self
        while next_node.right is not None:
            next_node = next_node.right
            yield next_node

    def re_attach(self):
        """
        reverse the pop
        :return:
        """
        if self.left is not None:
            self.left.right = self
        if self.right is not None:
            self.right.left = self

        if self.up is not None:
            self.up.down = self
        if self.down is not None:
            self.down.up = self

    def iter_up(self):
        next_node = self.up
        while next_node is not self:
            yield next_node
            next_node = next_node.up

    def iter_down(self):
        next_node = self.down
        while next_node is not self:
            yield next_node
            next_node = next_node.down

    def iter_left(self):
        next_node = self.left
        while next_node is not self:
            yield next_node
            next_node = next_node.left

    def iter_right(self):
        next_node = self.right
        while next_node is not self:
            yield next_node
            next_node = next_node.right

    def __str__(self):
        str_1 = str(self.text)
        return "[" + str_1 + "]"

    def __repr__(self):
        return self.__str__()


class matrix():
    def __init__(self, columns_str, rows, rows_str):
        self.header = QuadNode()
        header = self.header
        header.left = header
        header.right = header
        header.text = "HEAD"
        self.col_dict = {}

        for col_str in columns_str:
            q_node = QuadNode()
            q_node.right = header
            q_node.left = header.left
            header.left.right = q_node
            header.left = q_node
            q_node.text = col_str

            self.col_dict[col_str] = q_node

        self.row_dit = {}
        for i in range(len(rows)):
            self.row_dit[rows_str[i]] = self.add_row(rows[i], rows_str[i])

    def print(self):
        if not config.Debug:
            return
        for col in self.header.iter_left():
            print(str(col) + ":", end="")
            print(*tuple([r for r in col.iter_down()]))

    def print_r(self):
        if not config.Debug:
            return
        for row in self.header.iter_down():
            print(str(row) + ":", end=" ")

            print(*tuple([r.header for r in row.iter_right()]))
        print("◘" * 20)

    def add_row(self, row, row_text):
        first_node: QuadNode = QuadNode()
        first_node.text = row_text
        first_node.header = self.header
        first_node.up = self.header.up
        first_node.down = self.header
        self.header.up.down = first_node
        self.header.up = first_node
        for index in row:
            if index not in self.col_dict:
                continue
            col = self.col_dict[index]
            q_node = QuadNode()
            q_node.header = col
            q_node.text = row_text
            # last node
            q_node.up = col.up
            # connect last to the header
            q_node.down = col
            col.up.down = q_node
            col.up = q_node
            q_node.left = q_node
            q_node.right = q_node
            if first_node is not None:
                q_node.right = first_node
                q_node.left = first_node.left
                first_node.left.right = q_node
                first_node.left = q_node
            else:
                first_node = q_node
        return first_node

    def solve(self, depth=0, partial=[]):
        if self.header.right is self.header:
            return True, partial

        # select col
        # for col in self.header.right.iter_right():
        col: QuadNode = self.find_min_column()

        if col.down is col:
            return False, None
        self.cover_column(col)
        result, solution = False, None
        for row in col.iter_down():

            for column in row.iter_right():
                self.cover_column(column.header)

            result, solution = self.solve(depth + 1, partial + [row])
            for column in row.iter_left():
                self.uncover_column(column.header)

            if result:
                break

        self.uncover_column(col)
        return result, solution

    def find_min_column(self):
        min = 100000000000000000000
        min_col = self.header.right
        for col in self.header.iter_right():
            count = len([r for r in col.iter_down()])
            if count < min:
                min = count
                min_col = col
        return min_col

    def cover_row(self, row):
        # self.cover_column(row.header)
        for column in row.iter_right():
            self.cover_column(column.header)

    def cover_column(self, col):
        """

        :type col: QuadNode
        """
        if col is self.header:
            return

        col.left.right = col.right
        col.right.left = col.left
        for row in col.iter_down():
            for cell in row.iter_right():
                cell.up.down = cell.down
                cell.down.up = cell.up
        if config.Debug:
            print("Cover", str(col))

        self.print()
        self.print_r()

    def uncover_column(self, col):
        if col is self.header:
            return

        for row in col.iter_up():
            for cell in row.iter_left():
                cell.up.down = cell
                cell.down.up = cell

        col.right.left = col

        col.left.right = col
        if config.Debug:
            print("Uncover", str(col))
        self.print()
        self.print_r()


def sudoku(board):
    n = 9
    n_sq = 3
    co_cell = "Cell:"
    co_row = "Row:"
    co_collumn = "Col:"
    co_block = "Block:"
    # build_columns
    columns = []
    for r in range(n):
        for c in range(n):
            columns.append(f"{co_cell}{str(r)},{str(c)}")
    for r in range(n):
        for val in range(n):
            columns.append(f"{co_row}{str(r)},{str(val)}")
    for c in range(n):
        for val in range(n):
            columns.append(f"{co_collumn}{str(c)},{str(val)}")
    for r in range(n_sq):
        for c in range(n_sq):
            for val in range(n):
                columns.append(f"{co_block}{str(r)},{str(c)},{str(val)}")

    rows = []
    rows_str = []
    for r in range(n):
        for c in range(n):
            for val in range(n):
                new_row = []
                new_row.append(co_cell + str(r) + "," + str(c))
                new_row.append(co_row + str(r) + "," + str(val))
                new_row.append(co_collumn + str(c) + "," + str(val))
                new_row.append(co_block + str(r // n_sq) + "," + str(c // n_sq) + "," + str(val))
                rows.append(new_row)
                rows_str.append(("HEAD", r, c, val))

    dlx_m = matrix(columns, rows, rows_str)
    solution = []
    for r in range(n):
        for c in range(n):
            if board[r][c]:
                val = board[r][c]
                row = dlx_m.row_dit[("HEAD", r, c, val - 1)]
                dlx_m.cover_row(row)
                solution.append(row)

    dlx_m.print()
    dlx_m.print_r()

    result, solution = dlx_m.solve(0, solution)
    if not result:
        return False, None
    # print(solution)
    new_board = copy.deepcopy(board)
    for cell in solution:
        s, r, c, val = cell.text
        # assert not new_board[r][c] or new_board[r][c] == val
        new_board[r][c] = val + 1
    if config.Debug:
        print_board(board)
        print_board(new_board)
    return True, new_board


def solve(map):
    return sudoku(map)[1]
