from torch.utils.data import Dataset
import ARC_gym.utils.graphs as graphUtils
import ARC_gym.utils.tokenization as tok
import re
import numpy as np
import os,json
import random

class ARCEvaluationDataset(Dataset):
    '''
    This loads the ARC evaluation tasks.
    '''
    def __init__(self, base_dir="ARC/data/evaluation"):
        self.base_dir = base_dir
        self.arc_files = os.listdir(base_dir)
        self.all_tasks = []

        self.load_tasks()

    def arc_to_numpy(self, fpath):
        with open(fpath) as f:
            content = json.load(f)

        train_input_grids = []
        train_output_grids = []
        test_input_grids = []
        test_output_grids = []

        for g in content["train"]:
            inp = np.array(g["input"], dtype="int8")
            outp = np.array(g["output"], dtype="int8")

            inp = tuple(tuple(inner) for inner in inp)
            outp = tuple(tuple(inner) for inner in outp)

            train_input_grids.append(inp)
            train_output_grids.append(outp)

        for g in content["test"]:
            inp = np.array(g["input"], dtype="int8")
            outp = np.array(g["output"], dtype="int8")

            inp = tuple(tuple(inner) for inner in inp)
            outp = tuple(tuple(inner) for inner in outp)

            test_input_grids.append(inp)
            test_output_grids.append(outp)

        return (tok.tokenize_grid_batch(train_input_grids),
                tok.tokenize_grid_batch(train_output_grids),
                tok.tokenize_grid_batch(test_input_grids),
                tok.tokenize_grid_batch(test_output_grids))

    def load_tasks(self):
        for fname in self.arc_files:
            fpath = os.path.join(self.base_dir, fname)
            train_input, train_output, test_input, test_output = self.arc_to_numpy(fpath)

            self.all_tasks.append((train_input, train_output, test_input, test_output))

    def __len__(self):
        return len(self.all_tasks)

    def __getitem__(self, idx):
        S = {}

        S['xs'], S['ys'], S['xq'], S['yq'] = self.all_tasks[idx]
        S['task_desc'] = self.arc_files[idx]
        return S
