
class Task:

    def __init__(self, grid_dim_min=3, grid_dim_max=30):
        self.grid_dim_min = grid_dim_min
        self.grid_dim_max = grid_dim_max

    def generateInputs(self, k):
        input_grids = []

        for _ in range(k):
            input_grids.append(self._generateInput())

        return input_grids

    def generateOutputs(self, input_grids):
        output_grids = []
        for input_grid in input_grids:
            output_grids.append(self._generateOutput(input_grid))

        return output_grids
