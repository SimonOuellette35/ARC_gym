from tasks.Task import Task
import utils.grid_utils as grid_utils
import numpy as np

class MinColorFillBasic(Task):

    def __init__(self):
        super(MinColorFillBasic).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        return grid_utils.colorFillGrid(input_grid, min_key)

class MaxColorFillBasic(Task):

    def __init__(self):
        super(MaxColorFillBasic).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        return grid_utils.colorFillGrid(input_grid, max_key)

class MinColorFillAdvanced(Task):

    def __init__(self):
        super(MinColorFillAdvanced).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        tmp_grid = grid_utils.generateEmptyGrid(color=0, dim=color_count[min_key])
        return grid_utils.colorFillGrid(tmp_grid, min_key)

class MaxColorFillAdvanced(Task):

    def __init__(self):
        super(MaxColorFillAdvanced).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        tmp_grid = grid_utils.generateEmptyGrid(color=0, dim=color_count[max_key])
        return grid_utils.colorFillGrid(tmp_grid, max_key)

class MinColorFiltering(Task):

    def __init__(self):
        super(MinColorFiltering).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] == min_key:
                    output_grid[x, y] = min_key

        return output_grid


class MaxColorFiltering(Task):

    def __init__(self):
        super(MaxColorFiltering).__init__()

    def generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] == max_key:
                    output_grid[x, y] = max_key

        return output_grid

class EvenColorFiltering(Task):

    def __init__(self):
        super(EvenColorFiltering).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        even_colors = []
        for key, value in color_count.items():
            if value % 2 == 0:
                even_colors.append(key)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] in even_colors:
                    output_grid[x, y] = input_grid[x, y]

        return output_grid

class OddColorFiltering(Task):

    def __init__(self):
        super(OddColorFiltering).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        odd_colors = []
        for key, value in color_count.items():
            if value % 2 == 1:
                odd_colors.append(key)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] in odd_colors:
                    output_grid[x, y] = input_grid[x, y]

        return output_grid

class GreaterThanColorFiltering(Task):

    def __init__(self):
        super(GreaterThanColorFiltering).__init__()
        self.K = np.random.choice(np.arange(2, 10))

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        kept_colors = []
        for key, value in color_count.items():
            if value > self.K:
                kept_colors.append(key)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] in kept_colors:
                    output_grid[x, y] = input_grid[x, y]

        return output_grid

class LessThanColorFiltering(Task):

    def __init__(self):
        super(LessThanColorFiltering).__init__()
        self.K = np.random.choice(np.arange(2, 10))

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        kept_colors = []
        for key, value in color_count.items():
            if value < self.K:
                kept_colors.append(key)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] in kept_colors:
                    output_grid[x, y] = input_grid[x, y]

        return output_grid

class RowPixelCounting(Task):

    def __init__(self):
        super(RowPixelCounting).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        output_grid = np.zeros_like(input_grid)

        for color in range(1, 10):
            if color in color_count:
                num = color_count[color]
                for i in range(num):
                    output_grid[i, color] = color

        return output_grid

class ColumnPixelCounting(Task):

    def __init__(self):
        super(ColumnPixelCounting).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        output_grid = np.zeros_like(input_grid)

        for color in range(1, 10):
            if color in color_count:
                num = color_count[color]
                for i in range(num):
                    output_grid[color, i] = color

        return output_grid

class MaxColorWins(Task):

    def __init__(self):
        super(MaxColorWins).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] > 0:
                    output_grid[x, y] = max_key

        return output_grid

class MinColorWins(Task):

    def __init__(self):
        super(MinColorWins).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] > 0:
                    output_grid[x, y] = min_key

        return output_grid

class GridDimEvenOddFiltering(Task):

    def __init__(self):
        super(GridDimEvenOddFiltering).__init__()

    def generateInput(self):
        return grid_utils.generateRandomPixels()

    def generateOutput(self, input_grid):
        grid_dim = input_grid.shape[0]
        color_count = grid_utils.colorCount(input_grid)

        keep_colors = []
        if grid_dim % 2 == 0:
            # keep even colors
            for key, value in color_count.items():
                if value % 2 == 0:
                    keep_colors.append(key)
        else:
            # keep odd colors
            for key, value in color_count.items():
                if value % 2 == 1:
                    keep_colors.append(key)

        output_grid = np.zeros_like(input_grid)

        for x in range(input_grid.shape[0]):
            for y in range(input_grid.shape[1]):
                if input_grid[x, y] in keep_colors:
                    output_grid[x, y] = input_grid[x, y]

        return output_grid
