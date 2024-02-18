import ARC_gym.primitives as primitives
import random
import ARC_gym.utils.graphs as graphUtils
from ARC_gym.dataset import ARCGymDataset


class MetaDGP:

    def __init__(self, grid_size=5):
        self.grid_size=grid_size
        self.total_primitive_set = primitives.get_total_set()
        self.ITEM_token = 11
        self.IOSEP = 12

    def instantiateExperiment(self, trainN, testN, num_modules, comp_graph_dist, grid_dist,
                              k=5, max_graphs=500, augment_data=False):
        self.primitives = random.sample(self.total_primitive_set, num_modules)    # get a subset from the existing ones

        if comp_graph_dist['train']['num_nodes'][0] < 2 or comp_graph_dist['test']['num_nodes'][0] < 2:
            print("ERROR: num_nodes attribute in the computational graph characteristics must have a minimal value no smaller than 2.")
            exit(1)

        # Determine meta-train characteristics
        self.meta_train = {
            'num_nodes': comp_graph_dist['train']['num_nodes'],
            'num_pixels': grid_dist['train']['num_pixels'],
            'space_dist_x': grid_dist['train']['space_dist_x'],
            'space_dist_y': grid_dist['train']['space_dist_y']
        }

        self.meta_test = {
            'num_nodes': comp_graph_dist['test']['num_nodes'],
            'num_pixels': grid_dist['test']['num_pixels'],
            'space_dist_x': grid_dist['test']['space_dist_x'],
            'space_dist_y': grid_dist['test']['space_dist_y']
        }

        self.modules = [
            # input node
            {
                'model': None,
                'name': 'input',
                'inputEmbDim': 0,
                'outputEmbDim': 10,
                'inputType': '',
                'outputType': 'token'
            }
        ]
        for p in self.primitives:
            self.modules.append({
                'model': p[0],
                'name': p[1],
                'inputEmbDim': 10,
                'outputEmbDim': 10,
                'inputType': 'token',
                'outputType': 'token'
            })

        self.modules.append(
            # output node
            {
                'model': None,
                'name': 'output',
                'inputEmbDim': 10,
                'outputEmbDim': 0,
                'inputType': 'token',
                'outputType': ''
            }
        )

        print("Generating meta-training set tasks...")
        self.meta_train_tasks = MetaDGP.generateGraphs(self.meta_train, trainN, max_graphs, self.modules)

        print("Generating meta-test set tasks...")
        self.meta_test_tasks = MetaDGP.generateGraphs(self.meta_test, testN, max_graphs, self.modules)

        meta_train_dataset = ARCGymDataset(self.meta_train_tasks, self.modules, self.meta_train, k, self.grid_size, augment_data)
        meta_test_dataset = ARCGymDataset(self.meta_test_tasks, self.modules, self.meta_test, k, self.grid_size, augment_data)

        return meta_train_dataset, meta_test_dataset, self.meta_train_tasks, self.meta_test_tasks

    @staticmethod
    def generateGraphs(metadata, N, max_graphs, modules):
        G_list = []

        count = 0
        generated_graphs = graphUtils.generate_all_directed_graphs(len(modules), metadata, max_graphs)
        print("==> Length of generated graphs = ", len(generated_graphs))

        for graph in generated_graphs:
            ts = graphUtils.generate_topological_sorts(graph)[0]

            G_list.append((graph, ts))

            count += 1
            if count >= N:
                break

        return G_list
