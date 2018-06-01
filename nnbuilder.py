
import numpy
import neuralnetwork
import genome
import random

from neuralnetwork import find_best_match
from globalvars import BUILDING_CYCLES, SENSORS, ACTIONS

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def build_nn(nn, gen, building_cycles=BUILDING_CYCLES, verbose=False):

    for cycle in range(building_cycles):

        for neuron_idx in range(nn.nb_neurons):

            neuron = nn.neurons[neuron_idx]

            rule_idx = find_best_match(neuron, gen.get_predecessors())
            if rule_idx is None:
                continue
            rule = gen.rules[rule_idx]

            if verbose is True:
                print(nn.nb_neurons)
                print("index: "+str(neuron_idx))
                rule.print()

            # Create new neuron
            if rule.successor != [] and rule.extra != []:
                nn.create_neuron_from_model(neuron_idx, rule.extra, rule.codes)
                nn.rename_unique_neuron(neuron_idx, rule.successor)
            # Just rename
            elif rule.extra == []:
                nn.rename_unique_neuron(neuron_idx, rule.successor)
            # Extra rule: Set links between predecessor and extra - no renaming
            elif rule.successor == []:
                nn.update_links(neuron_idx, rule.extra, rule.codes)

            if verbose == True:
                print("Building neurons")
                print(neuron_idx)
                rule.print()
                print(nn.nb_neurons)
                print(nn.neurons)
                print(nn.links)

    nn.sort_sensors_actions(SENSORS, ACTIONS)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_build_nn():

    SENSORS = [[0], [1], [0, 0]]
    ACTIONS = [[0, 1], [1, 0], [1, 1]]
    random.seed(2)
    nn = neuralnetwork.Neuralnetwork()
    gen = genome.Genome()
    gen.make_clean_rules()
    build_nn(nn, gen, verbose=False)
    assert(nn.neurons == [[0], [1], [0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1], [0]]
    )
    assert(numpy.array_equal(
        nn.links,
        numpy.array([[0., 1., 0., 1., 0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 1., 1., 1.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 0., 1., 0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 1., 0., 1., 1., 1., 0., 0., 0.],
       [1., 1., 0., 1., 1., 1., 1., 0., 0.],
       [1., 1., 0., 1., 1., 1., 1., 1., 0.]])

    ))


# import matplotlib.pyplot as plt
# plt.ion()

#import networkx as nx

# def show_graph_with_labels(adjacency_matrix, mylabels):
    #rows, cols = numpy.where(adjacency_matrix == 1)
    #edges = zip(rows.tolist(), cols.tolist())
    #gr = nx.Graph()
    # gr.add_edges_from(edges)
    #nx.draw(gr, node_size=500, with_labels=True)
    # plt.show()

#labels = dict(zip(range(len(nn.neurons)),[str(i) for i in nn.neurons]))

#show_graph_with_labels(nn.links, labels)
