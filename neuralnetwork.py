

import numpy
import random
import pdb

from globalvars import MAX_HASH
from globalvars import MAX_NEURONS
from globalvars import NEURAL_NOISE

import rule
from weightnode import weightnode as weight_node


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def satisfies(name, rule):
    if len(rule) > len(name) or len(rule) == 0:
        return(False)
    for (idx, i) in enumerate(rule):
        if name[idx] != rule[idx]:
            return(False)
    return(True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_satisfies():
    assert(satisfies([], []) is False)
    assert(satisfies([0], [0]) is True)
    assert(satisfies([0, 1], [0]) is True)
    assert(satisfies([1], [0]) is False)
    assert(satisfies([1, 0], [0]) is False)
    assert(satisfies([0, 1, 0], [0, 1]) is True)
    assert(satisfies([1, 1, 0], [0, 1]) is False)
    assert(satisfies([1, 0], [1, 0, 1]) is False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Note: rule_names must be sorted, otherwise the match returned
#  is not necessarily the BEST


def find_best_match(name, rule_names):
    best = None
    for (idx, rule) in enumerate(rule_names):
        if satisfies(name, rule):
            best = idx
    return(best)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def bin_to_switch(b):
    return(2*b - 1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_find_best_match():
    rule_names = [[1],
                  [0, 1],
                  [1, 0],
                  [0, 1, 0],
                  [1, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 1, 0, 1],
                  [0, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1],
                  [0, 1, 0, 0, 1, 0, 0]]
    assert(find_best_match([1], rule_names) == 0)
    assert(find_best_match([0], rule_names) is None)
    assert(find_best_match([0, 1], rule_names) == 1)
    assert(find_best_match([0, 1, 1], rule_names) == 1)
    assert(find_best_match([0, 1, 0, 1], rule_names) == 3)
    assert(find_best_match([0, 1, 1, 1, 1], rule_names) == 1)
    assert(find_best_match([1, 0, 1, 1, 1], rule_names) == 4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Neuralnetwork:

    def __init__(self):
        self.nb_neurons = 3
        self._max_neurons = MAX_NEURONS
        self.neurons = [[0, 0, 1], [0, 0, 0], [0, 1]]

        # a list which maps (weight number) -> list of neurons indices
        # namehash[weight_node([])] returns [9]
        self.refresh_namehash()
        self.links = numpy.zeros([3, 3])
        self.create_link([0, 0, 1], [0, 0, 0], 1)
        self.create_link([0, 0, 0], [0, 1], 1)

        self.activations = numpy.zeros([3])

    # ------------------------------------------------ #

    def refresh_namehash(self):
        self.namehash = [[] for i in range(MAX_HASH)]
        for (neuron_idx, neuron) in enumerate(self.neurons):
            self.namehash[weight_node(neuron)].append(neuron_idx)

    # ------------------------------------------------ #

    def idx_to_name(self, idx):
        return(self.neurons[idx])

    # ------------------------------------------------ #

    def name_to_idx(self, name):
        return(self.namehash[weight_node(name)])

    # ------------------------------------------------ #

    def create_link(self, name1, name2, sign):
        from_idxs = self.name_to_idx(name1)
        to_idxs = self.name_to_idx(name2)
        for i in from_idxs:
            for j in to_idxs:
                self.links[j, i] = sign

    # ------------------------------------------------ #

    def update_links(self, idx_from, name_to, codes):
        idx_to_list = self.match_pattern(name_to)
        if codes[0] == 1:
            for idx_extra in idx_to_list:
                self.links[idx_extra, idx_from] = bin_to_switch(codes[1])
        if codes[2] == 1:
            for idx_extra in idx_to_list:
                self.links[idx_from, idx_extra] = bin_to_switch(codes[3])

    # ------------------------------------------------ #

    def match_pattern(self, name):
        return([i for (i, neuron) in enumerate(self.neurons)
                if satisfies(neuron, name)])

    # ------------------------------------------------ #

    def create_neuron(self, name):
        if name is None or name == []:
            return None
        if self.nb_neurons >= self._max_neurons:
            raise Exception("Too many neurons created.")
        self.neurons.append(name)
        self.nb_neurons = self.nb_neurons + 1
        newlinks = numpy.zeros([self.nb_neurons, self.nb_neurons])
        newlinks[:-1, :-1] = self.links[:, :]
        self.links = newlinks
        self.refresh_namehash()
        self.activations = numpy.zeros([self.nb_neurons])
        return(self.nb_neurons - 1)

    # ------------------------------------------------ #

    def rename_neurons(self, prev_name, next_name):
        get_idx = self.name_to_idx(prev_name)
        for idx in get_idx:
            self.neurons[idx] = next_name
        self.refresh_namehash()

    # ------------------------------------------------ #

    def rename_unique_neuron(self, idx, next_name):
        if next_name == []:
            self.delete_neuron(idx)
            return
        self.neurons[idx] = next_name
        self.refresh_namehash()

    # ------------------------------------------------ #

    def delete_neuron(self, idx):
        if idx < 0 or idx >= self.nb_neurons:
            raise Exception("Wrong index for suppression")
        if self.nb_neurons <= 1:
            return

        self.nb_neurons = self.nb_neurons - 1

        if idx == self.nb_neurons:
            self.neurons = self.neurons[:idx]
        else:
            self.neurons = self.neurons[:idx] + self.neurons[idx+1:]

        self.links = numpy.delete(self.links, [idx], axis=0)
        self.links = numpy.delete(self.links, [idx], axis=1)
        self.activations = numpy.delete(self.activations, [idx], axis=0)

        self.refresh_namehash()

    # ------------------------------------------------ #

    def nonempty_links_to(self, idx):
        truthlist = self.links[idx, :] != 0
        return(numpy.arange(self.nb_neurons)[truthlist])

    # ------------------------------------------------ #

    def nonempty_links_from(self, idx):
        truthlist = self.links[:, idx] != 0
        return(numpy.arange(self.nb_neurons)[truthlist])

    # ------------------------------------------------ #

    def create_neuron_from_model(self, idx_old, name_new, codes):
        if name_new == [] or name_new is None:
            return
        try:
            idx_new = self.create_neuron(name_new)
        except Exception:
            return

        linkto = self.nonempty_links_from(idx_old)
        for i in linkto:
            self.links[i, idx_new] = self.links[i, idx_old]

        linkfrom = self.nonempty_links_to(idx_old)
        for i in linkfrom:
            self.links[idx_new, i] = self.links[idx_old, i]

        if codes[0] == 1:
            self.links[idx_new, idx_old] = bin_to_switch(codes[1])
        if codes[2] == 1:
            self.links[idx_old, idx_new] = bin_to_switch(codes[3])

    # ------------------------------------------------ #

    def update_activations(self, sensors):
        # Set neuron values to activations
        self.activations[self.sensors_idx] = sensors[self.sensors_found]
        self.activations = numpy.clip(
            numpy.matmul(self.links, self.activations),
            -1.0, 1.0)
        # Get actions from neuron values
        actions = numpy.zeros(len(self.actions_found))
        actions[self.actions_found] = self.activations[self.actions_idx]

        # See if saving actions is needed later ?
        self.actions = actions
        return(actions)

    # ------------------------------------------------ #

    def sort_sensors_actions(self, sensors, actions):
        self.sensors_idx, self.sensors_found = \
            get_idx_list_for_sensors(self, sensors)
        self.actions_idx, self.actions_found = \
            get_idx_list_for_sensors(self, actions)

        perm = create_index_but(self.nb_neurons, self.sensors_idx +
                                self.actions_idx)

        self.sensors_idx = numpy.arange(len(self.sensors_idx))
        self.actions_idx = numpy.arange(len(self.sensors_idx), len(
            self.sensors_idx)+len(self.actions_idx))
        self.neurons = [self.neurons[i] for i in perm]
        self.activations = self.activations[perm]
        self.links = permute_connectivity(self.links, perm)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_sort_sensors_actions():
    nn = Neuralnetwork()
    nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 1])
    sensors = [[0], [0, 1]]
    actions = [[0, 0, 0], [1, 0]]
    nn.sort_sensors_actions(sensors, actions)
    assert(nn.sensors_idx == [0])
    assert(nn.sensors_found == [False, True])
    assert(nn.actions_idx == [1])
    assert(nn.actions_found == [True, False])
    assert(nn.neurons == [[0, 1], [0, 0, 0], [0, 0, 1], [0, 1]])
    assert(numpy.array_equal(
           nn.links,
           numpy.array([[0., 1., 0., 1.],
                        [0., 0., 1., 1.],
                        [0., 0., 0., 0.],
                        [0., 1., 1., 0.]])
           ))


def test_sort_sensors_actions2():
    nn = Neuralnetwork()
    nn.neurons = [[1, 1, 1], [0, 0, 0], [0, 1]]
    nn.links = numpy.array([[0., 0., 0.],
                            [1., 0, 0.],
                            [0., 1., 0.]]
                           )
    sensors = [[0], [1], [0, 0]]
    actions = [[0, 1], [1, 0]]
    nn.sort_sensors_actions(sensors, actions)

    assert(nn.sensors_found == [False, False, False])
    assert(nn.actions_idx == [0])
    assert(nn.actions_found == [True, False])
    assert(nn.neurons == [[0, 1], [1, 1, 1], [0, 0, 0]]
           )
    assert(numpy.array_equal(
           nn.links,
           array([[0., 0., 1.],
                  [0., 0., 0.],
                  [0., 1., 0.]])
           ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_update_activations():
    nn = Neuralnetwork()
    nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 0])
    sensors = [[0], [0, 0, 0]]
    actions = [[0, 1], [1, 0]]
    nn.sort_sensors_actions(sensors, actions)
    actionval = nn.update_activations(numpy.array([0, 1]))
    assert(numpy.array_equal(
        actionval, numpy.array([1., 0.])
    ))
    assert(numpy.array_equal(
        nn.activations,
        numpy.array([0., 1., 0., 1.])))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def create_index_but(size, fixed):
    if len(fixed) == 0:
        return(numpy.arange(size))
    fixed = numpy.array(fixed)
    temp = numpy.ones(size, dtype=bool)
    temp[fixed] = False
    index = numpy.arange(size)
    return(numpy.concatenate((fixed, index[temp])))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_create_index_but():
    assert(numpy.array_equal
           (
               create_index_but(10, [4, 6]),
               numpy.array([4, 6, 0, 1, 2, 3, 5, 7, 8, 9])
           ))
    assert(numpy.array_equal
           (
               create_index_but(3, []),
               numpy.array([0, 1, 2])
           ))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_Neuralnetwork():
    nn = Neuralnetwork()
    nn.create_neuron([0, 1, 1, 0])
    nn.create_link([0, 1, 1, 0], [0, 0, 1], -1)
    nn.create_neuron([0, 1, 1, 0])
    nn.create_link([0, 1, 1, 0], [0, 0, 0], -1)
    nn.create_link([1], [0, 0, 0], -1)
    assert(numpy.array_equal
           (nn.links, numpy.array([[0.,  0.,  0., -1.,  0.],
                                   [1.,  0.,  0., -1., -1.],
                                   [0.,  1.,  0.,  0.,  0.],
                                   [0.,  0.,  0.,  0.,  0.],

                                   [0.,  0.,  0.,  0.,  0.]])))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_Neuralnetwork_create_neuron_from_model():
    nn = Neuralnetwork()
    codes = [1, 0, 1, 0]
    neuron_idx = 1
    new_neuron = [0]
    nn.create_neuron_from_model(neuron_idx, new_neuron, codes)
    assert(numpy.array_equal(nn.links, numpy.array([[0.,  0.,  0.,  0.],
                                                    [1.,  0.,  0., -1.],
                                                    [0.,  1.,  0.,  1.],
                                                    [1., -1.,  0.,  0.]])))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def permute_connectivity(mat, perm):
    mat = mat[:, perm]
    return(mat[perm, :])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_permute_connectivity():
    mat = numpy.array([[i+j for i in range(3)] for j in range(10, 40, 10)])
    # array(
    # [[10, 11, 12],
    # [20, 21, 22],
    # [30, 31, 32]])
    perm = numpy.array([0, 2, 1])
    mat = permute_connectivity(mat, perm)
    assert(numpy.array_equal(mat, numpy.array(
        [[10, 12, 11],
         [30, 32, 31],
         [20, 22, 21]])))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_idx_list_for_sensors(nn, sensors):
    sensor_idx = []
    sensor_found = []
    for s in sensors:
        idxlist = nn.name_to_idx(s)
        if idxlist != []:
            # only take the FIRST neuron with sensor
            sensor_idx.append(idxlist[0])
            sensor_found.append(True)
        else:
            sensor_found.append(False)
    return(sensor_idx, sensor_found)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_get_idx_list_for_sensors():
    random.seed(2)
    nn = Neuralnetwork()
    nn.create_neuron_from_model(1, [0, 1], [1, 1, 1, 1])
    sensors = [[0], [0, 1]]
    actions = [[0, 0, 0], [1, 0]]
    # nn.neurons
    # >> [[0, 0, 1], [0, 0, 0], [0, 1], [0, 1]]
    assert(
        get_idx_list_for_sensors(nn, sensors)
        ==
        ([2], [False, True])
    )
    assert(
        get_idx_list_for_sensors(nn, actions)
        ==
        ([1], [True, False])
    )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Double fancy indexing works


def test_double_fancy_indexing():
    u = numpy.zeros([6])
    v = numpy.array([0, 3, 4])
    y = - numpy.arange(6)
    z = numpy.array([1, 3, 5])
    u[v] = y[z]
    assert(
        numpy.array_equal(u,
                          numpy.array([-1.,  0.,  0., -3., -5.,  0.])
                          ))
