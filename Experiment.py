from torch.utils.data import DataLoader
from tqdm import tqdm
import torch        # TODO: support JAX models? Keras?


class Experiment:

    # Note: the expected format of output for the batching_func is:
    # Each batch element is a dictionary:
    # 'xs': tensor of shape [batch_size, number of examples per task, GRID_DIM*GRID_DIM] the input grids of the demonstrations
    # 'ys': tensor of shape [batch_size, number of examples per task, GRID_DIM*GRID_DIM] the output grids of the demonstrations
    # 'xq': tensor of shape [batchsize, GRID_DIM*GRID_DIM] the input grid for the test pair.
    # 'yq': tensor of shape [batchsize, GRID_DIM*GRID_DIM] the output grid for the test pair.
    # 'label_seq' [Optional]: if a program induction task, this is the expected program solution as a token sequence.
    # 'task_desc': list of verbal task descriptions for each task

    # Note: the model must implement its own train_batch function that handles the specificities (e.g. the loss function, etc.).
    # Similarly, it needs a test_batch function that outputs an accuracy on the batch. This will probably require a wrapper to the
    # original model code.
    def __init__(self, exp_mode, model, data_generator, batching_func, num_epochs=1000, train_batch_size=1000, val_batch_size=100):
        '''
        Parameters:
        @param exp_mode: program induction vs grid transduction type of experiment. Value: 'induction' or 'transduction'.
        @param model: the PyTorch model to train/evaluate.
        @param data_generator: a DataGenerator that outputs a Dataset for training or test.
        @param batching_func: a lambda that manipulates the raw Dataset, perpares it for the specific model used.
        '''
        self.exp_mode = exp_mode
        self.model = model
        self.data_generator = data_generator
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.batching_func = batching_func
        self.num_epochs = num_epochs

    def train(self):
        training_dataset = self.data_generator.generate(mode='training')

        train_dataloader = DataLoader(training_dataset,
                                      batch_size=self.train_batch_size,
                                      collate_fn=lambda x:self.batching_func(x),
                                      shuffle=True)

        val_dataset = self.data_generator.generate(mode='validation')

        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.val_batch_size,
                                    collate_fn=lambda x:self.batching_func(x),
                                    shuffle=True)

        for epoch in tqdm(range(self.num_epochs), desc='Training epochs'):
            for batch_idx, train_batch in enumerate(train_dataloader):
                K = train_batch['xs'].shape[1]

                if self.exp_mode == 'induction':
                    loss = self.model.train_batch(train_batch['xs'], train_batch['ys'])
                else:
                    loss = self.model.train_batch(train_batch['xs'])

                print("Epoch %i, batch %i: training loss = %.2f" % (epoch+1, batch_idx+1, loss.cpu().data.numpy()))

            global_val_loss = 0.
            global_val_accuracy = 0.
            val_batch_count = 0.
            for batch_idx, val_batch in enumerate(val_dataloader):

                with torch.no_grad():
                    if self.exp_mode == 'induction':
                        val_loss = self.model.train_batch(val_batch['xs'], val_batch['ys'], training=False)
                        val_accuracy = self.model.test_batch(val_batch['xs'], val_batch['ys'])
                    else:
                        val_loss = self.model.train_batch(val_batch['xs'], training=False)
                        val_accuracy = self.model.test_batch(val_batch['xs'])

                global_val_loss += val_loss.cpu().data.numpy()
                global_val_accuracy += val_accuracy
                val_batch_count += 1

            global_val_loss /= val_batch_count
            global_val_accuracy /= val_batch_count
            print("==> Epoch complete: validation loss: %.4f, validation accuracy: %.2f %%" % (global_val_loss, global_val_accuracy * 100.))
                

    def test(self):
        test_dataset = self.data_generator.generate(mode='test')

        test_dataloader = DataLoader(test_dataset,
                                     batch_size=1,
                                     collate_fn=lambda x:self.batching_func(x),
                                     shuffle=False)

        global_accuracy = 0.
        batch_count = 0.
        for batch_idx, test_batch in enumerate(tqdm(test_dataloader, desc='Testing')):
            K = test_batch['xs'].shape[1]

            if self.exp_mode == 'induction':
                accuracy = self.model.test_batch(test_batch['xs'], test_batch['ys'])
            else:
                accuracy = self.model.test_batch(test_batch['xs'])

            global_accuracy += accuracy
            batch_count += 1.

        global_accuracy /= batch_count
        print("Test accuracy: %.2f %%" % (global_accuracy * 100.))