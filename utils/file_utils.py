import uuid
import json
import os

# The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:
#
# "train": demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
# "test": test input/output pairs. It is a list of "pairs" (typically 1 pair).
# A "pair" is a dictionary with two fields:
#
# "input": the input "grid" for the pair.
# "output": the output "grid" for the pair.
def save(task_list, output_path):
    for task_sample in task_list:
        support_inputs, support_outputs, query_inputs, query_outputs = task_sample

        train_pairs = []
        test_pairs = []
        for i in range(len(support_inputs)):
            pair = {
                "input": support_inputs[i].tolist(),
                "output": support_outputs[i].tolist()
            }

            train_pairs.append(pair)

        for i in range(len(query_inputs)):
            pair = {
                "input": query_inputs[i].tolist(),
                "output": query_outputs[i].tolist()
            }

            test_pairs.append(pair)

        json_obj = {
            "train": train_pairs,
            "test": test_pairs
        }

        # generate randomized filename
        filename = str(uuid.uuid4())[-12:]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open("%s/%s" % (output_path, filename), "w") as outfile:
            outfile.write(json.dumps(json_obj, indent=4))