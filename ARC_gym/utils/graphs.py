import networkx as nx
import itertools
import numpy as np
import random
import copy
from tqdm import tqdm


def generate_all_directed_graphs(num_modules, metadata, max_graphs, shuffle=True):
    generated_graphs = []
    counter = 0
    nodes = np.arange(1, num_modules+1)

    progress_bar = tqdm(total=max_graphs)

    for r in range(2, len(nodes) + 1):
        for sub_nodes in itertools.combinations(nodes, r):

            # Ensure both the start and end nodes are in the subset
            if nodes[0] not in sub_nodes or nodes[-1] not in sub_nodes:
                continue

            if len(sub_nodes) < metadata['num_nodes'][0] or len(sub_nodes) > metadata['num_nodes'][1]:
                continue

            possible_edges = list(itertools.permutations(sub_nodes, 2))

            G = nx.DiGraph()
            G.add_nodes_from(sub_nodes)

            for num_edges in range(1, len(possible_edges) + 1):
                for subset in itertools.combinations(possible_edges, num_edges):
                    G.clear_edges()
                    G.add_edges_from(subset)

                    # node 0 must be the starting point, and the last node must be the terminal node of the graph
                    if G.in_degree(nodes[0]) != 0 or G.out_degree(nodes[-1]) != 0 or G.in_degree(nodes[-1]) != 1 or G.out_degree(nodes[0]) == 0:
                        continue

                    if not all(G.in_degree(node) > 0 for node in G if node != nodes[0] and G.out_degree(node) > 0):
                        continue

                    # rule out cycles
                    if not nx.is_directed_acyclic_graph(G):
                        continue

                    # For every node, if it can be reached from the start node, check if the last node can be reached from it
                    if not all(nx.has_path(G, node, nodes[-1]) for node in G if nx.has_path(G, nodes[0], node)):
                        continue

                    # Check if the graph is weakly connected: don't allow disconnected subgraphs
                    if not nx.is_connected(G.to_undirected()):
                        continue

                    generated_graphs.append(copy.deepcopy(G))
                    progress_bar.update(1)
                    counter += 1
                    if len(generated_graphs) >= max_graphs:
                        if shuffle:
                            random.shuffle(generated_graphs)
                        progress_bar.close()
                        return generated_graphs

    if shuffle:
        random.shuffle(generated_graphs)
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
