from tasks.Task import Task
import utils.grid_utils as grid_utils
import numpy as np

class MinColorFillBasic(Task):

    def __init__(self):
        super(MinColorFillBasic).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        return grid_utils.colorFillGrid(input_grid, min_key)

class MaxColorFillBasic(Task):

    def __init__(self):
        super(MaxColorFillBasic).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        return grid_utils.colorFillGrid(input_grid, max_key)

class MinColorFillAdvanced(Task):

    def __init__(self):
        super(MinColorFillAdvanced).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        tmp_grid = grid_utils.generateEmptyGrid(color=0, dim=color_count[min_key])
        return grid_utils.colorFillGrid(tmp_grid, min_key)

class MaxColorFillAdvanced(Task):

    def __init__(self):
        super(MaxColorFillAdvanced).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        tmp_grid = grid_utils.generateEmptyGrid(color=0, dim=color_count[max_key])
        return grid_utils.colorFillGrid(tmp_grid, max_key)

class MinColorFiltering(Task):

    def __init__(self):
        super(MinColorFiltering).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        return grid_utils.generateRandomPixels()

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        return grid_utils.generateRandomPixels()

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        return grid_utils.generateRandomPixels()

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        return grid_utils.generateRandomPixels()

    def _generateOutput(self, input_grid):
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

class ColumnPixelCounting(Task):

    def __init__(self):
        super(ColumnPixelCounting).__init__()

    def _generateInput(self):
        return grid_utils.generateRandomPixels(max_pixels=15)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        output_grid = grid_utils.generateEmptyGrid(0, 15)

        for color in range(1, 10):
            if color in color_count:
                num = color_count[color]
                for i in range(num):
                    output_grid[i, color] = color

        return output_grid

class RowPixelCounting(Task):

    def __init__(self):
        super(RowPixelCounting).__init__()

    def _generateInput(self):
        return grid_utils.generateRandomPixels(max_pixels=15)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        output_grid = grid_utils.generateEmptyGrid(0, 15)

        for color in range(1, 10):
            if color in color_count:
                num = color_count[color]
                for i in range(num):
                    output_grid[color, i] = color

        return output_grid

class MaxColorWins(Task):

    def __init__(self):
        super(MaxColorWins).__init__()

    def _generateInput(self):
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

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels()
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
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

    def _generateInput(self):
        return grid_utils.generateRandomPixels()

    def _generateOutput(self, input_grid):
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

class EqualityTestMaxV1(Task):

    def __init__(self):
        super(EqualityTestMaxV1).__init__()

    def _generateInput(self):
        # TODO: must make sure that in half of the query examples, you have color_vals[0] != color_vals[1], and in the
        #  other half, you have the other case!
        #  This means the outer loop must not control the generation of batches of query examples, it must be managed
        #  within this task so that it can control the distribution!
        return grid_utils.generateRandomPixels(num_groups=2)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())

        if color_vals[0] != color_vals[1]:
            # fill background color with the min count color
            min_key = min(color_count, key=color_count.get)

            output_grid = grid_utils.generateEmptyGrid(min_key, input_grid.shape[0])

            # keep the pixels for the max count color only
            max_key = max(color_count, key=color_count.get)

            for x in range(input_grid.shape[0]):
                for y in range(input_grid.shape[1]):
                    if input_grid[x, y] == max_key:
                        output_grid[x, y] = input_grid[x, y]

            return output_grid
        else:
            return np.copy(input_grid)


class EqualityTestMinV1(Task):

    def __init__(self):
        super(EqualityTestMinV1).__init__()

    def _generateInput(self):
        # TODO: must make sure that in half of the query examples, you have color_vals[0] != color_vals[1], and in the
        #  other half, you have the other case!
        #  This means the outer loop must not control the generation of batches of query examples, it must be managed
        #  within this task so that it can control the distribution!
        return grid_utils.generateRandomPixels(num_groups=2)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())

        if color_vals[0] != color_vals[1]:
            # fill background color with the max count color
            max_key = max(color_count, key=color_count.get)

            output_grid = grid_utils.generateEmptyGrid(max_key, input_grid.shape[0])

            # keep the pixels for the min count color only
            min_key = min(color_count, key=color_count.get)

            for x in range(input_grid.shape[0]):
                for y in range(input_grid.shape[1]):
                    if input_grid[x, y] == min_key:
                        output_grid[x, y] = input_grid[x, y]

            return output_grid
        else:
            return np.copy(input_grid)

class EqualityTestV2(Task):

    def __init__(self):
        super(EqualityTestV2).__init__()
        self.equal_color = np.random.choice(np.arange(1, 10))
        self.unequal_color = np.random.choice(np.concatenate((np.arange(1, self.equal_color), np.arange(self.equal_color + 1, 10))))

    def _generateInput(self):
        # TODO: must make sure that in half of the query examples, you have color_vals[0] != color_vals[1], and in the
        #  other half, you have the other case!
        #  This means the outer loop must not control the generation of batches of query examples, it must be managed
        #  within this task so that it can control the distribution!
        return grid_utils.generateRandomPixels(num_groups=2)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())
        total_count = np.sum(color_vals)

        if color_vals[0] != color_vals[1]:
            return grid_utils.generateEmptyGrid(self.unequal_color, total_count)
        else:
            return grid_utils.generateEmptyGrid(self.equal_color, total_count)

class SumMin(Task):

    def __init__(self):
        super(SumMin).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        total_count = np.sum(list(color_count.values()))
        min_key = min(color_count, key=color_count.get)

        return grid_utils.generateEmptyGrid(min_key, total_count)

class SumMax(Task):

    def __init__(self):
        super(SumMax).__init__()

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        total_count = np.sum(list(color_count.values()))
        max_key = max(color_count, key=color_count.get)

        return grid_utils.generateEmptyGrid(max_key, total_count)

class DiffMin(Task):

    def __init__(self):
        super(DiffMin).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())
        difference = abs(color_vals[0] - color_vals[1])
        min_key = min(color_count, key=color_count.get)

        return grid_utils.generateEmptyGrid(min_key, difference)

class DiffMax(Task):

    def __init__(self):
        super(DiffMax).__init__()

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())
        difference = abs(color_vals[0] - color_vals[1])
        max_key = max(color_count, key=color_count.get)

        return grid_utils.generateEmptyGrid(max_key, difference)

class SumMinV2(Task):

    def __init__(self):
        super(SumMinV2).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            color_values2 = sorted(color_count.values())
            if len(color_values) > 2 and color_values[0] != color_values[1] and color_values2[0] != color_values2[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        min_key = min(color_count, key=color_count.get)
        max_key = max(color_count, key=color_count.get)

        sum_value = color_count[min_key] + color_count[max_key]

        return grid_utils.generateEmptyGrid(min_key, sum_value)


class SumMaxV2(Task):

    def __init__(self):
        super(SumMaxV2).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            color_values2 = sorted(color_count.values())
            if len(color_values) > 2 and color_values[0] != color_values[1] and color_values2[0] != color_values2[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        min_key = min(color_count, key=color_count.get)
        max_key = max(color_count, key=color_count.get)

        sum_value = color_count[min_key] + color_count[max_key]

        return grid_utils.generateEmptyGrid(max_key, sum_value)

class DiffMinV2(Task):

    def __init__(self):
        super(DiffMinV2).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            color_values2 = sorted(color_count.values())
            if len(color_values) > 2 and color_values[0] != color_values[1] and color_values2[0] != color_values2[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        min_key = min(color_count, key=color_count.get)
        max_key = max(color_count, key=color_count.get)

        diff_value = abs(color_count[min_key] - color_count[max_key])

        return grid_utils.generateEmptyGrid(min_key, diff_value)


class DiffMaxV2(Task):

    def __init__(self):
        super(DiffMaxV2).__init__()

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels=15)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            color_values2 = sorted(color_count.values())
            if len(color_values) > 2 and color_values[0] != color_values[1] and color_values2[0] != color_values2[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        min_key = min(color_count, key=color_count.get)
        max_key = max(color_count, key=color_count.get)

        diff_value = abs(color_count[min_key] - color_count[max_key])

        return grid_utils.generateEmptyGrid(max_key, diff_value)
