from tasks.Task import Task
import utils.grid_utils as grid_utils

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