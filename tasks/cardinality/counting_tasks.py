from tasks.Task import Task
import utils.grid_utils as grid_utils
import numpy as np
import random

class ColorOfMinBasic(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(ColorOfMinBasic, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values())
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        min_key = min(color_count, key=color_count.get)

        return grid_utils.colorFillGrid([[1]], min_key)

class ColorOfMaxBasic(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(ColorOfMaxBasic, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
            color_count = grid_utils.colorCount(tmp)
            color_values = sorted(color_count.values(), reverse=True)
            if len(color_values) == 1 or color_values[0] != color_values[1]:
                # solution unique!
                solution_unique = True

        return tmp

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)

        max_key = max(color_count, key=color_count.get)

        return grid_utils.colorFillGrid([[1]], max_key)

class MinColorFillBasic(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MinColorFillBasic, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MaxColorFillBasic, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MinColorFillAdvanced, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MaxColorFillAdvanced, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MinColorFiltering, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MaxColorFiltering, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(EvenColorFiltering, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        return grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(OddColorFiltering, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        return grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(GreaterThanColorFiltering, self).__init__(grid_dim_min, grid_dim_max)
        self.K = np.random.choice(np.arange(2, 10))

    def _generateInput(self):
        return grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(LessThanColorFiltering, self).__init__(grid_dim_min, grid_dim_max)
        self.K = np.random.choice(np.arange(2, 10))

    def _generateInput(self):
        return grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(ColumnPixelCounting, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        return grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(RowPixelCounting, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        return grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MaxColorWins, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(MinColorWins, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(GridDimEvenOddFiltering, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        return grid_utils.generateRandomPixels(grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(EqualityTestMaxV1, self).__init__(grid_dim_min, grid_dim_max)

    def generateInputs(self, k):
        input_grids = []

        num_equal = round(k/2)
        for _ in range(num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] == color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        for _ in range(k - num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] != color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self):
        return grid_utils.generateRandomPixels(num_groups=2, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(EqualityTestMinV1, self).__init__(grid_dim_min, grid_dim_max)

    def generateInputs(self, k):
        input_grids = []

        num_equal = round(k/2)
        for _ in range(num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] == color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        for _ in range(k - num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] != color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self):
        return grid_utils.generateRandomPixels(num_groups=2, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(EqualityTestV2, self).__init__(grid_dim_min, grid_dim_max)
        self.equal_color = np.random.choice(np.arange(1, 10))
        self.unequal_color = np.random.choice(np.concatenate((np.arange(1, self.equal_color), np.arange(self.equal_color + 1, 10))))

    def generateInputs(self, k):
        input_grids = []

        num_equal = round(k/2)
        for _ in range(num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] == color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        for _ in range(k - num_equal):
            ok = False
            while not ok:
                tmp_grid = self._generateInput()

                color_count = grid_utils.colorCount(tmp_grid)
                color_vals = list(color_count.values())

                if color_vals[0] != color_vals[1]:
                    ok = True

            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self):
        return grid_utils.generateRandomPixels(num_groups=2, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

    def _generateOutput(self, input_grid):
        color_count = grid_utils.colorCount(input_grid)
        color_vals = list(color_count.values())
        total_count = np.sum(color_vals)

        if color_vals[0] != color_vals[1]:
            return grid_utils.generateEmptyGrid(self.unequal_color, total_count)
        else:
            return grid_utils.generateEmptyGrid(self.equal_color, total_count)

class SumMin(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(SumMin, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(SumMax, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(DiffMin, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(DiffMax, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the max color is unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(num_groups=2, max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(SumMinV2, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(SumMaxV2, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(DiffMinV2, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        super(DiffMaxV2, self).__init__(grid_dim_min, grid_dim_max)

    def _generateInput(self):
        # only allow generating grid for which the min AND max colors are unique
        solution_unique = False
        while not solution_unique:
            tmp = grid_utils.generateRandomPixels(max_pixels_per_color=15, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)
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

class BasicCountingV1(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30, num_px_min=0, num_px_max=10):
        super(BasicCountingV1, self).__init__(grid_dim_min, grid_dim_max)
        self.num_px_min = num_px_min
        self.num_px_max = num_px_max

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            num_px = np.random.choice(np.arange(self.num_px_min, self.num_px_max))
            tmp_grid = self._generateInput(num_px)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               grid_dim_min=self.grid_dim_min,
                                               grid_dim_max=self.grid_dim_max)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.pixelCount(input_grid)
        pixel_count_binary = grid_utils.decimalToBinary(pixel_count, dim=9)
        pixel_count_binary = pixel_count_binary[::-1]
        return np.reshape(pixel_count_binary, [3, 3])

class BasicCountingV2(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30, num_px_min=1, num_px_max=10):
        super(BasicCountingV2, self).__init__(grid_dim_min, grid_dim_max)
        self.num_px_min = num_px_min
        self.num_px_max = num_px_max

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            num_px = np.random.choice(np.arange(self.num_px_min, self.num_px_max))
            tmp_grid = self._generateInput(num_px)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt, grid_dim_min=self.grid_dim_min, grid_dim_max=self.grid_dim_max)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.pixelCount(input_grid)
        return grid_utils.generateEmptyGrid(pixel_count, 1)

class BasicCountingV3(Task):

    def __init__(self, grid_dim_min=3, grid_dim_max=30, num_px_min=1, num_px_max=10):
        super(BasicCountingV3, self).__init__(grid_dim_min, grid_dim_max)
        self.num_px_min = num_px_min
        self.num_px_max = num_px_max

    def generateInputs(self, k):
        input_grids = []
        for _ in range(k):
            num_px = np.random.choice(np.arange(self.num_px_min, self.num_px_max))
            tmp_grid = self._generateInput(num_px)
            input_grids.append(tmp_grid)

        random.shuffle(input_grids)
        return input_grids

    def _generateInput(self, mpt):
        return grid_utils.generateRandomPixels(max_pixels_total=mpt,
                                               grid_dim_min=self.grid_dim_min,
                                               grid_dim_max=self.grid_dim_max,
                                               sparsity=0.8)

    def _generateOutput(self, input_grid):
        pixel_count = grid_utils.perColorPixelCount(input_grid)
        grid_px_count = np.reshape(pixel_count, [3, 3])
        return grid_px_count
