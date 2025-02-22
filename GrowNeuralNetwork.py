from typing import Optional
import numpy
import neuralnetwork
import genome
import random

from neuralnetwork import find_best_match
from globalvars import BUILDING_CYCLES, SENSORS, ACTIONS
from neuralnetwork import Neuralnetwork
from genome import Genome

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def GrowNeuralNetwork(
    nn: Neuralnetwork,
    gen: Genome,
    building_cycles: int = BUILDING_CYCLES,
    verbose: bool = False
) -> None:

    for cycle in range(building_cycles):

        for neuron_idx in range(nn.nb_neurons):

            neuron = nn.neurons[neuron_idx]

            rule_idx = find_best_match(neuron, gen.get_predecessors())
            if rule_idx is None:
                continue
            rule = gen.rules[rule_idx]

            if verbose:
                print(nn.nb_neurons)
                print(f"index: {neuron_idx}")
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

            if verbose:
                print("Building neurons")
                print(neuron_idx)
                rule.print()
                print(nn.nb_neurons)
                print(nn.neurons)
                print(nn.links)

    nn.sort_sensors_actions(SENSORS, ACTIONS)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
