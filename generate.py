import argparse
import tasks.cardinality.task_list as cardinality_tasks
import utils.file_utils as file_utils

# Usage: generate.py -t <task> -l <level> -o <output_path> -n <number>
# Where:
#   <task> is a task group name (directories under tasks) or "all" for all tasks. Default: "all"
#   <level> is the curriculum level, such as: 0, 1, 2 or "all" for all curriculum levels. Default: "all"
#   <output_path> is the path where the generated data will be saved. Default: "."
#   <number> is the number of task samples to generate, for both training and evaluation folders. Default: 100
parser = argparse.ArgumentParser(
                    prog='ARC_gym',
                    description='ARC gym: a data generator for the ARC challenge')

parser.add_argument('-t', '--task', default='all')
parser.add_argument('-l', '--level', default='all')
parser.add_argument('-o', '--output', default='.')
parser.add_argument('-n', '--number', default='100')

args = parser.parse_args()

if args.task == 'all' or args.task == 'cardinality':
    if args.level == 'all':
        curriculum_level = None
    else:
        curriculum_level = args.level

    task_list = cardinality_tasks.generate(curriculum_level, args.number)
    file_utils.save(task_list)