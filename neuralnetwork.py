import numpy
import random
import pdb
import numpy as np
from typing import List, Tuple, Optional, Union
import numpy.typing as npt

from globalvars import MAX_HASH, MAX_NEURONS, NEURAL_NOISE, FACTOR_MATRIX

import rule
from weightnode import weightnode as weight_node


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def satisfies(name: List[int], rule: List[int]) -> bool:
    if len(rule) > len(name) or len(rule) == 0:
        return(False)
    for (idx, i) in enumerate(rule):
        if name[idx] != rule[idx]:
            return(False)
    return(True)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def find_best_match(name: List[int], rule_names: List[List[int]]) -> Optional[int]:
    best = None
    for (idx, rule) in enumerate(rule_names):
        if satisfies(name, rule):
            best = idx
    return(best)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def bin_to_switch(b: int) -> int:
    return(2*b - 1)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Neuralnetwork:

    def __init__(self) -> None:
        self.nb_neurons: int = 3
        self._max_neurons: int = MAX_NEURONS
        self.neurons: List[List[int]] = [[0, 0, 1], [0, 0, 0], [0, 1]]

        # a list which maps (weight number) -> list of neurons indices
        # namehash[weight_node([])] returns [9]
        self.refresh_namehash()
        self.links: npt.NDArray[np.float64] = numpy.zeros([3, 3])
        self.create_link([0, 0, 1], [0, 0, 0], 1)
        self.create_link([0, 0, 0], [0, 1], 1)

        self.activations: npt.NDArray[np.float64] = numpy.zeros([3])

    # ------------------------------------------------ #

    def refresh_namehash(self) -> None:
        self.namehash: List[List[int]] = [[] for i in range(MAX_HASH)]
        for (neuron_idx, neuron) in enumerate(self.neurons):
            self.namehash[weight_node(neuron)].append(neuron_idx)

    # ------------------------------------------------ #

    def idx_to_name(self, idx: int) -> List[int]:
        return(self.neurons[idx])

    # ------------------------------------------------ #

    def name_to_idx(self, name: List[int]) -> List[int]:
        return(self.namehash[weight_node(name)])

    # ------------------------------------------------ #

    def create_link(self, name1: List[int], name2: List[int], sign: int) -> None:
        from_idxs = self.name_to_idx(name1)
        to_idxs = self.name_to_idx(name2)
        for i in from_idxs:
            for j in to_idxs:
                self.links[j, i] = sign

    # ------------------------------------------------ #

    def update_links(self, idx_from: int, name_to: List[int], codes: List[int]) -> None:
        idx_to_list = self.match_pattern(name_to)
        if codes[0] == 1:
            for idx_extra in idx_to_list:
                self.links[idx_extra, idx_from] = bin_to_switch(codes[1])
        if codes[2] == 1:
            for idx_extra in idx_to_list:
                self.links[idx_from, idx_extra] = bin_to_switch(codes[3])

    # ------------------------------------------------ #

    def match_pattern(self, name: List[int]) -> List[int]:
        return([i for (i, neuron) in enumerate(self.neurons)
                if satisfies(neuron, name)])

    # ------------------------------------------------ #

    def create_neuron(self, name: Optional[List[int]]) -> Optional[int]:
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

    def rename_neurons(self, prev_name: List[int], next_name: List[int]) -> None:
        get_idx = self.name_to_idx(prev_name)
        for idx in get_idx:
            self.neurons[idx] = next_name
        self.refresh_namehash()

    # ------------------------------------------------ #

    def rename_unique_neuron(self, idx: int, next_name: List[int]) -> None:
        if next_name == []:
            self.delete_neuron(idx)
            return
        self.neurons[idx] = next_name
        self.refresh_namehash()

    # ------------------------------------------------ #

    def delete_neuron(self, idx: int) -> None:
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

    def nonempty_links_to(self, idx: int) -> npt.NDArray[np.int64]:
        truthlist = self.links[idx, :] != 0
        return(numpy.arange(self.nb_neurons)[truthlist])

    # ------------------------------------------------ #

    def nonempty_links_from(self, idx: int) -> npt.NDArray[np.int64]:
        truthlist = self.links[:, idx] != 0
        return(numpy.arange(self.nb_neurons)[truthlist])

    # ------------------------------------------------ #

    def create_neuron_from_model(self, idx_old: int, name_new: List[int], codes: List[int]) -> None:
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

    def update_activations(self, sensors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Set neuron values to activations
        self.activations[self.sensors_idx] = self.activations[self.sensors_idx] + \
            sensors[self.sensors_found]
        self.activations = numpy.clip(
            numpy.matmul(self.links, self.activations)*FACTOR_MATRIX,
            -1.0, 1.0)
        # Get actions from neuron values
        actions = numpy.zeros(len(self.actions_found))
        actions[self.actions_found] = self.activations[self.actions_idx]

        # See if saving actions is needed later ?
        self.actions = actions
        return(actions)

    # ------------------------------------------------ #

    def sort_sensors_actions(self, sensors: List[List[int]], actions: List[List[int]]) -> None:
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


def create_index_but(size: int, fixed: List[int]) -> npt.NDArray[np.int64]:
    if len(fixed) == 0:
        return(numpy.arange(size))
    fixed = numpy.array(fixed)
    temp = numpy.ones(size, dtype=bool)
    temp[fixed] = False
    index = numpy.arange(size)
    return(numpy.concatenate((fixed, index[temp])))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def permute_connectivity(mat: npt.NDArray[np.float64], perm: List[int]) -> npt.NDArray[np.float64]:
    mat = mat[:, perm]
    return(mat[perm, :])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_idx_list_for_sensors(nn: Neuralnetwork, sensors: List[List[int]]) -> Tuple[List[int], List[bool]]:
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
