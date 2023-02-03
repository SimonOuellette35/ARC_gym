import argparse
import tasks.cardinality.task_list as cardinality_tasks
import utils.file_utils as file_utils
import random
import numpy as np

# Usage: generate.py -t <task> -l <level> -o <output_path> -n <number> -s <split>
# Where:
#   <task> is a task group name (directories under tasks) or "all" for all tasks. Default: "all"
#   <level> is the curriculum level, such as: 0, 1, 2 or "all" for all curriculum levels. Default: "all"
#   <output_path> is the path where the generated data will be saved. Default: "."
#   <number> is the number of task samples to generate, for both training and evaluation. Default: 100
#   <split> is the percentage of task samples kept for training (the rest being test tasks). Default: 0.8

parser = argparse.ArgumentParser(
                    prog='ARC_gym',
                    description='ARC gym: a data generator for the ARC challenge')

parser.add_argument('-t', '--task', default='all')
parser.add_argument('-l', '--level', default='all')
parser.add_argument('-o', '--output', default='.')
parser.add_argument('-n', '--number', default='100')
parser.add_argument('-s', '--split', default=0.8)

args = parser.parse_args()

def generate(task_list, level, number):

    def generateSample(k=3):

        if level is None:
            level_idx = np.random.choice(np.arange(len(task_list.keys())))
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

def meta_split(task_list, train_pct):

    def split_level(level_task_list):
        num_train_tasks = round(train_pct * len(level_task_list))
        random.shuffle(level_task_list)
        return level_task_list[:num_train_tasks], level_task_list[num_train_tasks:]

    meta_train = {}
    meta_test = {}
    for level, level_task_list in task_list.items():
        train_list, test_list = split_level(level_task_list)
        meta_train[level] = train_list
        meta_test[level] = test_list

    return meta_train, meta_test

if args.task == 'all' or args.task == 'cardinality':
    if args.level == 'all':
        curriculum_level = None
    else:
        curriculum_level = args.level

    # manage meta-training/meta-test split (based on split value)
    task_list = cardinality_tasks.task_list

    training_task_dict, test_task_dict = meta_split(task_list, float(args.split))

    training_samples = generate(training_task_dict, curriculum_level, int(args.number))
    file_utils.save(training_samples, "%s/training" % args.output)

    test_samples = generate(test_task_dict, curriculum_level, int(args.number))
    file_utils.save(test_samples, "%s/test" % args.output)