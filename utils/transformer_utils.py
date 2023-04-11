import numpy as np
import torch

bins = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 400]
forward_max = 400

def pad(l):
  for b in bins + [forward_max]:
    if b >= l: return b
  raise IndexError("Length %s longer than max length %s" % (l, forward_max))

def grid_to_sequence(grid):
    idx = 0
    seq = np.zeros((grid.shape[0] * grid.shape[1]))
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            seq[idx] = grid[x, y]
            idx += 1

    return seq.astype(int)

PAD_VAL = -1

# This class generates transformer-adapted training/test sequences for 1 specific task instance
class UTTaskDataGenerator:
    """The base class for generating problem input/output pairs"""
    # nclass = 50
    # name = '<unknown task>'
    # taskid = 0
    # height = None
    # min_length = 1

    def __init__(self, task, input_grid_dim=10, output_grid_dim=1):
        self.task = task
        self.input_grid_dim = input_grid_dim
        self.output_grid_dim = output_grid_dim

    def is_valid_length(self, l):
        """Can this problem have instances of length l?"""
        return True

    def rand_pair(self, length):
        if (self.input_grid_dim * self.input_grid_dim) > length:
            print("length = ", length)
            print("self.grid_dim = ", self.input_grid_dim)
            print("Error: your grid dimensionality is too large for the specified sequence length.")
            exit(-1)

        example_input = self.task.generateInputs(1)
        example_output = self.task.generateOutputs(example_input)

        # inp is a sequence of tokens, of length <= length (if < length, it will be padded in rand_pair_padded)
        inp = grid_to_sequence(example_input[0])

        # res is a sequence of tokens corresponding to the expected output.
        res = grid_to_sequence(example_output[0])

        return inp, res

    def rand_pair_padded(self, length, rand_length):
        """Construct a random data pair, then pad the inputs to a valid size."""
        pad_length = pad(length)
        l = length
        inp, outp = self.rand_pair(l)
        inp = np.array(inp)

        padding_func = lambda x: np.pad(x, [(0, 0)] * (len(x.shape) - 1) +
                                        [(0, pad_length - x.shape[-1])],
                                        'constant', constant_values=PAD_VAL)
        inp, outp = padding_func(inp), padding_func(np.array(outp))

        assert inp.shape[-1] == pad_length, outp.shape[-1] == pad_length
        return inp, outp

    def get_batch(self, length, batch_size, rand_length = True):
        """Construct a complete batch of problem instances"""
        inps, outps = [], []
        for _ in range(batch_size):
            inp, outp = self.rand_pair(length)
            inps.append(inp)
            outps.append(outp)

        inp = np.stack(inps, 0)
        outp = np.stack(outps, 0)
        src_padding_mask, tgt_padding_maks = self.generate_padding_masks(inp, outp)
        return torch.tensor(inp), torch.tensor(outp), torch.tensor(src_padding_mask), torch.tensor(tgt_padding_maks)

    @staticmethod
    def generate_padding_masks(source, target):
        src_padding_mask = source == PAD_VAL
        tgt_padding_maks = target == PAD_VAL
        return src_padding_mask, tgt_padding_maks

    def _initialize(self, nclass):
        self.nclass = nclass

    def __repr__(self):
        return "<%s name='%s' taskid=%s>" % (
        self.__class__.__name__, self.name, self.taskid)

