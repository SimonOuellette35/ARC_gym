import networkx as nx
import itertools
import numpy as np
import random
import copy
from tqdm import tqdm


def generate_all_directed_graphs(num_modules, metadata, max_graphs, use_tqdm=True, shuffle=True):

    generated_graphs = []
    nodes = np.arange(1, num_modules+1)

    if use_tqdm:
        progress_bar = tqdm(total=max_graphs)

    for _ in range(max_graphs):
        G = nx.DiGraph()

        num_nodes = np.random.choice(np.arange(metadata['num_nodes'][0], metadata['num_nodes'][1]+1))
        edges = []
        for node_idx in range(num_nodes):
            edge = []

            # from edge
            if node_idx == 0:
                edge.append(1)
            else:
                edge.append(edges[-1][1])

            # to edge
            if node_idx == num_nodes-1:
                edge.append(num_modules)
            else:
                ok = False
                while not ok:
                    rnd_node = np.random.choice(np.arange(2, num_modules))
                    ok = True

                    for prev_edge in edges:
                        if rnd_node in prev_edge:
                            ok = False
                            break

                edge.append(rnd_node)

            edges.append(edge)

        G.add_edges_from(edges)
        generated_graphs.append(copy.deepcopy(G))

        if use_tqdm:
            progress_bar.update(1)

    if shuffle:
        random.shuffle(generated_graphs)
    if use_tqdm:
        progress_bar.close()

    return generated_graphs

def get_desc(ts, modules):
    desc = ''
    for edge in ts:
        from_module = modules[edge[0] - 1]['name']
        to_module = modules[edge[1] - 1]['name']

        desc += '(%s ==> %s)' % (from_module, to_module)

    return desc

def getInputNodes(toposort, node):
    input_nodes = []
    print("Input Nodes: G.edges() = ", list(toposort))
    for tmp_edge in toposort:
        if tmp_edge[1] == node:
            if tmp_edge[0] not in input_nodes:
                input_nodes.append(tmp_edge[0])

    return input_nodes

def getOutputNodes(toposort, node):
    output_nodes = []
    print("Output Nodes: G.edges() = ", list(toposort))
    for tmp_edge in toposort:
        if tmp_edge[0] == node:
            if tmp_edge[1] not in output_nodes:
                output_nodes.append(tmp_edge[1])

    return output_nodes

def executeCompGraph(G, input, modules):
    graph, toposort = G
    nodes = list(graph.nodes)

    results = [None] * (len(modules) - 1)
    results[0] = [input]

    def execute(n, inp_params):
        model = modules[n - 1]['model']
        if len(inp_params) == 1:
            if model is None:
                return inp_params[0]

            #print("Calling module %s" % (modules[n - 1]['name']))
            return model(inp_params[0])
        else:
            output_grid = np.zeros_like(inp_params[0])
            for inp in inp_params:
                output_grid += inp

            sum_inp = output_grid % 10
            if model is None:
                return sum_inp

            #print("Calling module %s" % (modules[n-1]['name']))
            return model(sum_inp)

    for edge in toposort:
        #print("==> edge = ", edge)
        inp_params = results[edge[0]-1]

        to_node = edge[1]

        tmp_out = execute(to_node, inp_params)
        if to_node == nodes[-1]:
            return tmp_out

        if results[to_node - 1] is None:
            results[to_node - 1] = [tmp_out]
        else:
            # A concatenation is happening
            results[to_node - 1].append(tmp_out)

    return results[-1]


def get_concatenation_points(graph):
    # 1) get the adjacency matrix, identify concatenation points.
    adj_matrix = nx.to_numpy_array(graph)
    node_list = list(graph.nodes)

    concat_points = []
    for col_idx in range(adj_matrix.shape[1]):
        if np.sum(adj_matrix[:, col_idx]) > 1:
            concat_node = node_list[col_idx]
            concat_points.append(concat_node)

    return concat_points

def generate_topological_sorts(graph):
    concat_points = get_concatenation_points(graph)

    # 2) generate a topological sort
    toposort = list(nx.topological_sort(nx.line_graph(graph)))

    # 3) going through the topological sort, branch out the orderings everytime a concatenation point is met.
    def branch(concat_point, toposorts):
        def get_permutations(ts, cat_pt):
            indices = []
            for idx in range(len(ts)):
                if ts[idx][1] == cat_pt:
                    indices.append(idx)

            # permutations over these indices
            perm_indices = itertools.permutations(indices)

            def generate_new_sort(ts, perm_indices, orig_indices):
                output_ts = np.copy(ts)

                def get_range(pi):
                    from_idx = orig_indices[0]
                    for idx in orig_indices:
                        if pi == idx:
                            return from_idx, pi+1
                        from_idx = idx+1

                from_orig_idx = orig_indices[0]
                for pi in perm_indices:
                    from_idx, to_idx = get_range(pi)
                    length = to_idx - from_idx

                    output_ts[from_orig_idx: from_orig_idx+length] = ts[from_idx:to_idx]
                    from_orig_idx += length

                return output_ts

            output_list = []
            for perm in perm_indices:
                # must NOT add already existing permutation.
                if list(perm) != indices:
                    # generate corresponding topological sort and add to output list.
                    tmp_ts = generate_new_sort(ts, perm, indices)
                    output_list.append(tmp_ts)

            return output_list

        tmp_sorts = []
        for ts in toposorts:
            new_toposorts = get_permutations(ts, concat_point)
            for nt in new_toposorts:
                tmp_sorts.append(nt)

        return tmp_sorts

    toposort_variants = np.array([toposort])
    for cat_pt in concat_points:
        tmp_variants = branch(cat_pt, toposort_variants)
        toposort_variants = np.concatenate((toposort_variants, tmp_variants))

    return toposort_variants
