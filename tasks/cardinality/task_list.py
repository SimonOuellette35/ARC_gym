import numpy as np
import counting_tasks

# TODO: list of all task generation objects
task_list = {
    0: [counting_tasks.MinColorFillBasic(),
        counting_tasks.MaxColorFillBasic(),
        counting_tasks.MinColorFillAdvanced(),
        counting_tasks.MaxColorFillAdvanced()],
    1: [],
    2: []
}

NUM_CURRICULUM_LEVELS = 3

def generate(level, number):

    def generateSample(k=3):

        if level is None:
            level_idx = np.random.choice(np.arange(NUM_CURRICULUM_LEVELS))
        else:
            level_idx = int(level)

        task_idx = np.random.choice(np.arange(len(task_list[level_idx])))

        support_inputs = []
        support_outputs = []
        query_inputs = []
        query_outputs = []

        for _ in range(k):
            input_grid = task_list[level_idx][task_idx].generateInput()
            output_grid = task_list[level_idx][task_idx].generateOutput(input_grid)

            support_inputs.append(input_grid)
            support_outputs.append(output_grid)

        for _ in range(k):
            input_grid = task_list[level_idx][task_idx].generateInput()
            output_grid = task_list[level_idx][task_idx].generateOutput(input_grid)

            query_inputs.append(input_grid)
            query_outputs.append(output_grid)

        return support_inputs, support_outputs, query_inputs, query_outputs

    task_samples = []
    for _ in range(number):
        sample = generateSample()

        task_samples.append(sample)

    return task_samples
