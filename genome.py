from typing import List, Optional, Any
import random
from dataclasses import dataclass

from rule import Rule
from globalvars import ADN_LENGTH, MAX_BINARY_SIZE_FOR_NEURONS, MAX_HASH
from weightnode import weightnode as weight_node

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def truncate(neuron):
    if len(neuron) > MAX_BINARY_SIZE_FOR_NEURONS:
        return(neuron[:MAX_BINARY_SIZE_FOR_NEURONS])
    return(neuron)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def argsort(seq):
    # http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    # by unutbu
    return sorted(range(len(seq)),  key=seq.__getitem__)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def prune_list_by_index(lis, ind):
    return([lis[i] for i in ind])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def sort_unique(weight):
    inew = argsort(weight)
    iprime = [inew[0]]
    for idx, i in enumerate(inew):
        if idx > 0:
            if weight[i] > prev:
                iprime.append(i)
        prev = weight[i]
    return(iprime)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@dataclass
class Genome:
    adn: List[int]
    rules: List[Rule]
    adn_length: int = ADN_LENGTH

    def __init__(self, auto_initialize: bool = True) -> None:
        if auto_initialize:
            self.random_adn()
        else:
            self.adn = []
        self.rules = []

    # ------------------------------------------------ #

    def make_rules(self) -> None:
        local_adn = self.adn
        while len(local_adn) > 0:
            nextrule = Rule()
            local_adn = nextrule.build_from_adn(local_adn)
            
            if nextrule.isRule():
                self.rules.append(nextrule)

    # ------------------------------------------------ #

    def random_adn(self) -> None:
        self.adn = [random.randint(0, 1) for _ in range(self.adn_length)]

    # ------------------------------------------------ #

    def make_clean_rules(self) -> None:
        self.make_rules()
        self.clear_rules()

    # ------------------------------------------------ #

    def get_predecessors(self) -> List[str]:
        return [rule.predecessor for rule in self.rules]

    # ------------------------------------------------ #

    def show_rules(self) -> None:
        for rule in self.rules:
            print("------------")
            rule.print()

    # ------------------------------------------------ #

    def clear_rules(self) -> None:
        self.rules = [
            rule for rule in self.rules 
            if len(rule.predecessor) > 0 and 
               (len(rule.successor) + len(rule.extra) > 0)
        ]

        for rule in self.rules:
            rule.predecessor = truncate(rule.predecessor)
            rule.successor = truncate(rule.successor)
            rule.extra = truncate(rule.extra)

        pred_weight = [weight_node(rule.predecessor) for rule in self.rules]
        iuniq = sort_unique(pred_weight)
        self.rules = prune_list_by_index(self.rules, iuniq)
