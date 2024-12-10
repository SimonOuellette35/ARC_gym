from ARC_gym.MetaDGP import MetaDGP
from ARC_gym.utils.batching import make_mlc_batch
from torch.utils.data import DataLoader
import ARC_gym.utils.metrics as metrics
import ARC_gym.utils.visualization as viz
import numpy as np

train_batch_size = 100
test_batch_size = 10
NUM_TRAIN_TASKS = 500
NUM_TEST_TASKS = 10
MAX_GRAPHS = 250
GRID_DIM = 6

# Here you control the "out-of-distributionness" of the test set w.r.t to the training set, based on the number of
# primitives used to compose the tasks. In the test set, here we generate tasks of 5 to 6 primitives, while in the
# training set we generate tasks of 3 to 4 primitives. This "guarantees" that test tasks will be structurally 
# distinct from the training tasks. You can change that to have a partial, or even total overlap.
comp_graph_dist = {
    'train': {
        'num_nodes': [3, 4]
    },
    'test': {
        'num_nodes': [5, 6]     # Out-of-distribution meta-test set
    }
}

# This controls the distribution of the randomly generated grids for training and test. Here, we pick the same
# distribution. But you can change the range of generated pixels per grid, and even the spatial distribution
# of those pixels (probabilistically).
grid_dist = {
    'train': {
        'num_pixels': [1, 5],
        'space_dist_x': np.ones(GRID_DIM) / float(GRID_DIM),
        'space_dist_y': np.ones(GRID_DIM) / float(GRID_DIM)
    },
    'test': {
        'num_pixels': [1, 5],
        'space_dist_x': np.ones(GRID_DIM) / float(GRID_DIM),
        'space_dist_y': np.ones(GRID_DIM) / float(GRID_DIM)
    }
}

# Instantiate an experiment and corresponding data loaders
dgp = MetaDGP(grid_size=GRID_DIM)
meta_train_dataset, meta_test_dataset, meta_train_tasks, meta_test_tasks = dgp.instantiateExperiment(
    trainN=NUM_TRAIN_TASKS,
    testN=NUM_TEST_TASKS,
    num_modules=12,
    comp_graph_dist=comp_graph_dist,
    grid_dist=grid_dist,
    max_graphs=MAX_GRAPHS)

meta_train_dataloader = DataLoader( meta_train_dataset,
                                    batch_size=train_batch_size,
                                    collate_fn=lambda x:make_mlc_batch(x), # This will generate "in-context learning"
                                                                            # type of sequences to use in a seq-to-seq
                                                                            # model such as a Transformer, for example.
                                    shuffle=False)
meta_test_dataloader = DataLoader(  meta_test_dataset,
                                    batch_size=test_batch_size,
                                    collate_fn=lambda x:make_mlc_batch(x),
                                    shuffle=False)

# Measure to which extent the meta-dataset is out-of-distribution. Here we expect an ODDness of 1 due to how the
# computational graph characteristics have been selected. That is, the test set is entirely out-of-distribution w.r.t
# the training set.
OODness = metrics.quantify_comp_graph_OOD(meta_train_dataset, meta_train_tasks,
                                          meta_test_dataset, meta_test_tasks, dgp.modules)

print("==> Meta-dataset OODness = ", OODness)

# Optionally: visualize the generated tasks.
#viz.draw_batch(meta_train_dataloader, 5, 4)
viz.draw_batch(meta_train_dataset[0], 5, 4)

# The main training/evaluation loop. Insert your algorithm here.
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    print("==> Epoch #%i" % epoch)
    for batch_idx, train_batch in enumerate(meta_train_dataloader):
        print("Train batch input sequences shape: ", train_batch['xq+xs+ys_padded'].shape)
        print("First batch sample has input sequence: ", train_batch['xq+xs+ys_padded'][0].cpu().data.numpy())

        print("First batch sample has target sequence: ", train_batch['yq_padded'][0].cpu().data.numpy())
        # TODO: your model training/evaluation/etc.