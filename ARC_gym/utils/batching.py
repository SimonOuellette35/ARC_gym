import torch
import numpy as np

def make_mlc_batch(batch, ITEM_token=11, IOSEP_token=12, k=5):

    batch_size = len(batch)

    mybatch = {}
    mybatch['batch_size'] = batch_size
    mybatch['task_desc'] = []
    mybatch['xq'], mybatch['yq'], mybatch['xs'], mybatch['ys'] = [], [], [], []

    mybatch['q_idx'] = []  # index of which episode each query belongs to
    mybatch['s_idx'] = []  # index of which episode each support belongs to
    mybatch['xq+xs+ys'] = []

    for task_idx in range(batch_size):
        task_idx = np.random.choice(np.arange(len(batch)))

        mybatch['xq'].append(np.reshape(batch[task_idx]['xq'], [-1]))
        mybatch['yq'].append(torch.from_numpy(np.reshape(batch[task_idx]['yq'], [-1])))
        mybatch['xs'].append(np.reshape(batch[task_idx]['xs'], [k, -1]))
        mybatch['ys'].append(np.reshape(batch[task_idx]['ys'], [k, -1]))

        myquery = np.reshape(batch[task_idx]['xq'], [-1])

        xq_xs_ys = []
        for k_idx in range(k):
            tmp = np.concatenate((myquery,
                                       [ITEM_token],
                                       np.reshape(batch[task_idx]['xs'][k_idx], [-1]),
                                       [IOSEP_token],
                                       np.reshape(batch[task_idx]['ys'][k_idx], [-1])), axis=0)

            xq_xs_ys.append(tmp)

        mybatch['xq+xs+ys'].append(torch.from_numpy(np.array(xq_xs_ys)))
        mybatch['q_idx'].append(task_idx * torch.ones(1, dtype=torch.int))
        mybatch['s_idx'].append(task_idx * torch.ones(k, dtype=torch.int))
        mybatch['task_desc'].append(batch[task_idx]['task_desc'])

    mybatch['q_idx'] = torch.cat(mybatch['q_idx'], dim=0)
    mybatch['s_idx'] = torch.cat(mybatch['s_idx'], dim=0)

    mybatch['xq+xs+ys_padded'] = torch.stack(mybatch['xq+xs+ys'])  # m*nq x ns x max_len_q_pairs
    mybatch['yq_padded'] = torch.stack(mybatch['yq'])  # m*nq x ns x max_len_q_pairs

    return mybatch

# This creates batches in essentially the same format as the dataset itself, i.e:
#  'xs': support example inputs as a tensor [batch_size x k x grid_size * grid_size]
#  'ys': support example outputs as a tensor [batch_size x k x grid_size * grid_size]
#  'xq': query example input as a tensor [batch_size x grid_size * grid_size]
#  'yq': query example output as a tensor [batch_size x grid_size * grid_size]
#  'task_desc': a string that describes the task [batch_size x string length]
def make_gridcoder_batch(batch):
    mybatch = {}

    k = batch[0]['xs'].shape[0]
    batch_size = len(batch)

    mybatch['xs'], mybatch['ys'], mybatch['xq'], mybatch['yq'] = [], [], [], []
    mybatch['task_desc'] = []

    for task_idx in range(batch_size):
        mybatch['xq'].append(torch.from_numpy(np.reshape(batch[task_idx]['xq'], [-1])))
        mybatch['yq'].append(torch.from_numpy(np.reshape(batch[task_idx]['yq'], [-1])))
        mybatch['xs'].append(torch.from_numpy(np.reshape(batch[task_idx]['xs'], [k, -1])))
        mybatch['ys'].append(torch.from_numpy(np.reshape(batch[task_idx]['ys'], [k, -1])))

    mybatch['xq'] = torch.stack(mybatch['xq'], dim=0)
    mybatch['yq'] = torch.stack(mybatch['yq'], dim=0)
    mybatch['xs'] = torch.stack(mybatch['xs'], dim=0)
    mybatch['ys'] = torch.stack(mybatch['ys'], dim=0)
    return mybatch