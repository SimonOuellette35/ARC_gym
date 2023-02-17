import argparse
import utils.file_utils as file_utils
from SampleGenerator import SampleGenerator

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


if args.task == 'all' or args.task == 'cardinality':

    if args.level == 'all':
        curriculum_level = None
    else:
        curriculum_level = args.level

    generator = SampleGenerator(args.task, args.level, args.number, args.split, args.output)

    training_samples, test_samples = generator.generateDataset()
    file_utils.save(training_samples, "%s/training" % args.output)

    if len(test_samples) > 0:
        file_utils.save(test_samples, "%s/test" % args.output)
