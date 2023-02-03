
# The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:
#
# "train": demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
# "test": test input/output pairs. It is a list of "pairs" (typically 1 pair).
# A "pair" is a dictionary with two fields:
#
# "input": the input "grid" for the pair.
# "output": the output "grid" for the pair.
def save(task_list):
    # TODO