import numpy as np
import ARC_gym.utils.graphs as graphUtils

def exact_match(a_batch, b_batch):
    for a, b in zip(a_batch, b_batch):
        if (a != b).any():
            return False

    return True

def quantify_comp_graph_OOD(train_dataset, train_task, test_dataset, test_task, modules):

    print("Quantifying OOD-ness over %i test tasks." % (len(test_dataset)))
    oodness = len(test_dataset)
    for test_task_idx in range(len(test_dataset)):
        for train_task_idx in range(len(train_dataset)):
            train_sample = train_dataset[train_task_idx]
            test_sample = test_dataset[test_task_idx]
            grid_dim = train_sample['xs'].shape[1]

            x = np.concatenate((train_sample['xs'],
                                train_sample['xq'],
                                test_sample['xs'],
                                test_sample['xq']), axis=0)

            x = np.reshape(x, [x.shape[0], grid_dim, grid_dim])

            train_y = []
            test_y = []
            for x_sample in x:
                tmp_train_y = graphUtils.executeCompGraph(train_task[train_task_idx], x_sample, modules)
                tmp_test_y = graphUtils.executeCompGraph(test_task[test_task_idx], x_sample, modules)

                train_y.append(tmp_train_y)
                test_y.append(tmp_test_y)

            # If there is any training task for which we get an exact match on outputs (given the same inputs),
            # it is for all intents and purposes the same task -- so we have an overlap between training and test tasks.
            if exact_match(train_y, test_y):
                oodness -= 1
                break

    return oodness / float(len(test_dataset))

def checkAccuracy(preds, ground_truths):
    acc = 0
    for idx in range(len(preds)):
        if (preds[idx] == ground_truths[idx]).all():
            acc += 1

    if acc == len(preds):
        return True
    else:
        return False

def calculateAccuracy(preds, ground_truths):
    acc = 0.
    for batch_idx in range(preds.shape[0]):
        if checkAccuracy(preds[batch_idx], ground_truths[batch_idx]):
            acc += 1.

    return acc / float(preds.shape[0])
