import tasks.cardinality.task_list as cardinality_tasks
import random
import numpy as np

class SampleGenerator:

    def __init__(self, task_group="all", level="all", number=100, split=0.8):
        self.task_list = cardinality_tasks.task_list
        self.split = split
        self.number = number
        self.level = level
        self.task_group = task_group

    def meta_split(self, task_list, train_pct):

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

    # Generates <number> task samples drawn from the provided task_list's <level> curriculum level.
    # Each task sample will have k support examples, and k query examples.
    # See tasks.cardinality.task_list for a task_list example.
    def generateTaskSamples(self, task_list, level, number, k=5):

        def generateSample(k):

            if level is None:
                level_idx = np.random.choice(np.arange(len(task_list.keys())))
            else:
                level_idx = int(level)

            task_idx = np.random.choice(np.arange(len(task_list[level_idx])))

            task_instance = task_list[level_idx][task_idx]()

            support_inputs = task_instance.generateInputs(k)
            support_outputs = task_instance.generateOutputs(support_inputs)

            query_inputs = task_instance.generateInputs(k)
            query_outputs = task_instance.generateOutputs(query_inputs)

            return support_inputs, support_outputs, query_inputs, query_outputs

        task_samples = []
        for _ in range(number):
            sample = generateSample(k)

            task_samples.append(sample)

        return task_samples

    # Generates a training and test dataset based on the pre-specified parameters.
    def generateDataset(self):

        # manage meta-training/meta-test split (based on split value)
        training_task_dict, test_task_dict = self.meta_split(self.task_list, float(self.split))

        # TODO: temporary until we implement all curriculum levels:
        curriculum_level = 0

        training_samples = self.generateTaskSamples(training_task_dict, curriculum_level, int(self.number))

        # Here we check if the split ratio resulted in using only training data (i.e., no test data), which will be
        # the case if split = 1.
        use_test_data = True
        if self.level == 'all':
            if len(list(test_task_dict[0])) == 0:
                use_test_data = False
        else:
            if len(list(test_task_dict[int(curriculum_level)])) == 0:
                use_test_data = False

        test_samples = []
        if use_test_data:
            test_samples = self.generateTaskSamples(test_task_dict, curriculum_level, int(self.number))

        return training_samples, test_samples
