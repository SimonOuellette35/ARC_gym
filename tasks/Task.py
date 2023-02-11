
class Task:

    def __init__(self):
        pass

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
